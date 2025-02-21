# from copy import deepcopy

# from loguru import logger
# from torch import Tensor, nn, no_grad, zeros_like
# from torch import device as torch_device
# from torch.nn import BCEWithLogitsLoss
# from torch.optim import Adam, Optimizer
# from torch.utils.data import DataLoader

# import wandb
# from src.helper_function import metalearning_binary_target_changer


# class Model(nn.Module):
#     def __init__(
#         self,
#         nodes_per_layer: list[int],
#         activations_per_layer: list[nn.Module],
#     ):
#         super().__init__()
#         # All layers should have an activation
#         assert len(nodes_per_layer) - 1 == len(activations_per_layer)

#         # Create neural net
#         self.layers = []
#         for i in range(len(nodes_per_layer) - 1):
#             self.layers.append(nn.Linear(nodes_per_layer[i], nodes_per_layer[i + 1]))
#             self.layers.append(activations_per_layer[i])

#         self.layers = nn.Sequential(*self.layers)

#     def forward(self, x: Tensor) -> Tensor:
#         return self.layers(x)


# class Reptile:  # Assumes binary classifier for now
#     def __init__(
#         self,
#         model: nn.Module,
#         *,
#         train_n_gradient_steps: int,
#         eval_n_gradient_steps: int,
#         device: torch_device,
#         meta_optimizer: Optimizer,
#         inner_lr: float,
#         outer_lr: float,
#         k_shot: int,
#         loss_function: nn.Module = None,
#     ):
#         assert (
#             next(model.parameters()).device == device
#         ), "Model parameters are not on the specified device"

#         self.model = model
#         self.train_n_gradient_steps = train_n_gradient_steps
#         self.eval_n_gradient_steps = eval_n_gradient_steps
#         # self.loss_fn = loss_function  # TODO hypere_params?
#         self.loss_fn = loss_function or BCEWithLogitsLoss()
#         self.device = device
#         self.outer_optimizer = meta_optimizer
#         self.inner_lr = inner_lr
#         self.outer_lr = outer_lr
#         self.k_shot = k_shot

#     def _set_learning_rate(self, lr):
#         for param_group in self.outer_optimizer.param_groups:
#             param_group["lr"] = lr

#     def evaluate(self, dataloader: DataLoader, epoch: int, lr: float):
#         """Evaluate the current model on the full dataset.

#         This differs from training, as it does the inner loop on a support set but evaluates on a query set,
#         that is usually the rest of the data for the classes in the task.


#         """
#         global_model = self.model

#         accuracies = []
#         for X, y in dataloader:
#             # get support/query data
#             y = metalearning_binary_target_changer(y)
#             X, y = X.to(self.device), y.to(self.device)
#             X_support = X[: self.k_shot * 2, :]
#             y_support = y[: self.k_shot * 2]
#             X_query = X[self.k_shot * 2 :, :]
#             y_query = y[self.k_shot * 2 :]

#             self.inner_loop(X_support, y_support, lr, True, X_query, y_query, accuracies)

#         mean_accuracy = sum(accuracies) / len(accuracies)
#         logger.info(f"Evaluation after epoch {epoch}: Accuracy = {mean_accuracy:.2f}")

#     def inner_loop(self, X_support, y_support, lr, eval=False, X_query=None, y_query=None, accuracies=None):
#         # Store the current parameters of the model
#         state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}

#         self.model.train()
#         # Create inner loop optimizer with β1=0 as per paper
#         inner_optimizer = Adam(
#             self.model.parameters(), lr=lr, betas=(0.0, 0.999)
#         )  # TODO hyper-params

#         # Inner loop training
#         for k in range(self.train_n_gradient_steps):
#             inner_optimizer.zero_grad()
#             outputs = self.model(X_support)
#             loss = self.loss_fn(outputs.squeeze(), y_support)
#             loss.backward()
#             inner_optimizer.step()

#         if eval:
#             assert X_query is not None and y_query is not None and accuracies is not None, "Missing arguments for evaluation"
#             # Evaluate model on query set
#             self.model.eval()
#             with no_grad():
#                 outputs = self.model(X_query)
#                 predictions = (outputs > 0).float()
#                 n_correct = (predictions.squeeze() == y_query).sum().item()
#                 task_accuracy = n_correct / y_query.size(0)
#                 accuracies.append(task_accuracy)

#         # Get updated parameters
#         updated_state_dict = {k: v.detach() for k, v in self.model.state_dict().items()}

#         # Restore original parameters
#         self.model.load_state_dict(state_dict)

#         return updated_state_dict

#     def fit(
#         self,
#         *,
#         train_dataloader: DataLoader,
#         n_epochs: int,
#         n_parallel_tasks: int,
#         evaluate_train: bool = False,
#         val_dataloader: DataLoader = None,
#     ):
#         if evaluate_train:
#             assert wandb.run is not None

#         self.model.train()  # Paramters of the model are called phi

#         # We want to go through the whole dataset in each epoch
#         n_iters = n_epochs * (len(train_dataloader) // n_parallel_tasks + 1)
#         logger.info(
#             f"Starting training with {n_iters} iterations and {n_parallel_tasks} parallel tasks."
#         )
#         epoch_count = 0
#         # Calculate learning-rates
#         new_outer_lr = self.outer_lr * (1.0 - (epoch_count/float(n_epochs))**0.5)
#         self._set_learning_rate(new_outer_lr)
#         new_inner_lr = self.inner_lr * (1.0 - (epoch_count/float(n_epochs))**0.5)

#         # Evluation before starting
#         if evaluate_train:
#             self.evaluate(val_dataloader, epoch_count, new_inner_lr)

#         task_iterator = iter(train_dataloader)

#         for i in range(n_iters):
#             # Initialize an accumulator for meta gradients (same shape as parameters)
#             meta_gradients = {
#                 name: zeros_like(param.data)
#                 for name, param in self.model.named_parameters()
#             }

#             # n_parallel_batch models take self.train_n_gradient_steps, each for different tasks
#             # updating a copy of global_model to parameters phi'. This gives a difference
#             # phi - phi', contributing to model_param_deltas.
#             for j in range(n_parallel_tasks):
#                 # get a task
#                 try:
#                     X, y = next(task_iterator)
#                 except StopIteration:
#                     # DataLoader is exhausted, meaning one epoch is done.
#                     epoch_count += 1
#                     # Calculate new learning-rates
#                     new_outer_lr = self.outer_lr * (1.0 - (epoch_count/float(n_epochs))**0.5)
#                     self._set_learning_rate(new_outer_lr)
#                     new_inner_lr = self.inner_lr * (1.0 - (epoch_count/float(n_epochs))**0.5)

#                     logger.info(
#                         f"Epoch {epoch_count} complete at iteration {i+1}/{n_iters} with {n_parallel_tasks} parallel tasks and {len(train_dataloader)} total tasks. Reinitializing DataLoader for next epoch."
#                     )
#                     if evaluate_train:
#                         self.evaluate(val_dataloader, epoch_count, new_inner_lr)

#                     # prevents the model from training for more on some tasks than others
#                     if epoch_count == n_epochs:
#                         logger.info(
#                             f"Ending training as {n_epochs} epochs have been completed."
#                         )
#                         return
#                     # get the task
#                     task_iterator = iter(train_dataloader)
#                     X, y = next(task_iterator)

#                 self.model.train()
#                 y = metalearning_binary_target_changer(y)
#                 X, y = X.to(self.device), y.to(self.device)
#                 updated_state_dict = self.inner_loop(X, y, new_inner_lr)

#                 # Compute the pseudo-gradient: difference between the global model and the adapted model.
#                 # According to the paper, one may define the meta-gradient as (global - adapted)/outer_lr.
#                 with no_grad():
#                     for name, global_param in self.model.named_parameters():
#                         # Accumulate the difference over tasks.
#                         meta_gradients[name] += (
#                             global_param.detach() - updated_state_dict[name]
#                         )

#             print(
#                 f"Meta-update iteration {i+1}/{n_iters} complete: processed {n_parallel_tasks} parallel tasks."
#             )

#             # Average the accumulated meta gradients and scale by inner_lr.
#             self.outer_optimizer.zero_grad()
#             for name, param in self.model.named_parameters():
#                 print(param)
#                 meta_grad = meta_gradients[name] / n_parallel_tasks
#                 meta_grad = meta_grad / self.inner_lr

#                 # Assign the pseudo-gradient for the meta optimizer
#                 param.grad = meta_grad.clone()
#                 print(f"params: {param.grad}")
#             # Perform the meta-update using the outer optimizer.
#             self.outer_optimizer.step()


from copy import deepcopy

from loguru import logger
from torch import nn, no_grad, zeros_like
from torch import device as torch_device
from torch import cat as torch_cat
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader

import wandb
from src.helper_function import metalearning_binary_target_changer
from src.scoring.metalearning_scoring_fn import compute_metrics


class Reptile:  # Assumes binary classifier for now
    def __init__(
        self,
        model: nn.Module,
        *,
        train_n_gradient_steps: int,
        eval_n_gradient_steps: int,
        device: torch_device,
        inner_lr: float,
        outer_lr: float,
        inner_rl_reduction_factor: float,
        betas: tuple[float, float] = None,
        k_shot: int = None,
        loss_fn: nn.Module = None,
    ):
        assert (
            next(model.parameters()).device == device
        ), "Model parameters are not on the specified device"

        self.model = model
        self.train_n_gradient_steps = train_n_gradient_steps
        self.eval_n_gradient_steps = eval_n_gradient_steps
        # self.loss_fn = loss_function  # TODO hypere_params?
        self.loss_fn = loss_fn or BCEWithLogitsLoss()
        self.device = device
        self.inner_lr = inner_lr
        self.inner_rl_reduction_factor = inner_rl_reduction_factor
        self.outer_lr = outer_lr
        self.betas = betas or (0.0, 0.999)
        self.k_shot = k_shot

    def _set_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def evaluate(self, dataloader: DataLoader, epoch: int, lr: float):
        """Evaluate the current model on the full dataset.

        This differs from training, as it does the inner loop on a support set but evaluates on a query set,
        that is usually the rest of the data for the classes in the task.
        """
        global_model = self.model

        meta_test_error = 0.0
        predictions_all = []
        targets_all = []
        for X, y in dataloader:
            # get support/query data
            X, y = X.to(self.device), y.to(self.device)
            X_support = X[: self.k_shot * 2, :]
            y_support = y[: self.k_shot * 2]
            X_query = X[self.k_shot * 2 :, :]
            y_query = y[self.k_shot * 2 :]

            # copy model and setup inner_optimizer
            copied_model = deepcopy(global_model)
            copied_model.train()
            inner_optimizer = SGD(
                copied_model.parameters(),
                lr=lr,  # betas=self.betas
            )

            lr = self.inner_lr
            for k in range(self.eval_n_gradient_steps):
                loss = self.loss_fn(copied_model(X_support).squeeze(), y_support)
                loss.backward()
                inner_optimizer.step()
                inner_optimizer.zero_grad()
                lr /= self.inner_rl_reduction_factor
                self._set_learning_rate(inner_optimizer, lr)
                

            copied_model.eval()
            with no_grad():
                predictions = copied_model(X_query)
                evaluation_error = self.loss_fn(predictions, y_query)
                meta_test_error += evaluation_error.item()
                predictions_all.append(predictions)
                targets_all.append(y_query)

        meta_test_error /= len(dataloader)
        big_preds = torch_cat(predictions_all, dim=0)
        big_targets = torch_cat(targets_all, dim=0)
        final_scores = compute_metrics(big_preds, big_targets)
        wandb.log(
            {
                "val/loss": meta_test_error,
                "val/accuracy": final_scores["accuracy"],
                "val/f1": final_scores["f1"],
                "val/precision": final_scores["precision"],
                "val/recall": final_scores["recall"],
                "val/roc_auc": final_scores["roc_auc"],
                "epoch": epoch,
            },
        )

        logger.info(f"Evaluation after epoch {epoch}: Loss = {meta_test_error:.2f}")
        logger.info(
            f"Accuracy = {final_scores['accuracy']:.2f}, F1 = {final_scores['f1']:.2f}, Precision = {final_scores['precision']:.2f}, Recall = {final_scores['recall']:.2f}, ROC-AUC = {final_scores['roc_auc']:.2f}"
        )

    def fit(
        self,
        *,
        train_dataloader: DataLoader,
        n_epochs: int,
        n_parallel_tasks: int,
        evaluate_train: bool = False,
        val_dataloader: DataLoader = None,
    ):
        if evaluate_train:
            assert (
                wandb.run is not None
                and val_dataloader is not None
                and self.k_shot is not None
            ), "Missing arguments for evaluation"

        global_model = self.model  # nice reference. Parameters are called phi here
        global_model.train()

        # we want each epoch to go through the whole dataset
        n_iters = n_epochs * (len(train_dataloader) // n_parallel_tasks + 1)
        logger.info(
            f"Starting training with {n_iters} iterations and {n_parallel_tasks} parallel tasks."
        )
        epoch_count = 0
        # new_outer_lr = self.outer_lr * (1.0 - (epoch_count / float(n_epochs)) ** 0.5)
        # self._set_learning_rate(new_outer_lr)
        if evaluate_train:
            self.evaluate(val_dataloader, epoch_count, self.inner_lr)
        task_iterator = iter(train_dataloader)

        outer_optimizer = SGD(self.model.parameters(), self.outer_lr)

        for i in range(n_iters):
            meta_train_error = 0.0
            outputs_all = []
            targets_all = []
            # Initialize an accumulator for meta gradients (same shape as parameters)
            meta_gradients = {
                name: zeros_like(param.data)
                for name, param in global_model.named_parameters()
            }

            # n_parallel_batch models take self.train_n_gradient_steps, each for different tasks
            # updating a copy of global_model to parameters phi'. This gives a difference
            # phi - phi', contributing to model_param_deltas.
            for j in range(n_parallel_tasks):
                # get a task
                try:
                    X, y = next(task_iterator)
                except StopIteration:
                    # DataLoader is exhausted, meaning one epoch is done.
                    epoch_count += 1
                    # Update lr
                    # new_outer_lr = self.outer_lr * (
                    #     1.0 - (epoch_count / float(n_epochs)) ** 0.5
                    # )
                    # self._set_learning_rate(new_outer_lr)
                    logger.info(
                        f"Epoch {epoch_count} complete at iteration {i+1}/{n_iters} with {n_parallel_tasks} parallel tasks and {len(train_dataloader)} total tasks. Reinitializing DataLoader for next epoch."
                    )
                    if evaluate_train:
                        self.evaluate(val_dataloader, epoch_count, self.inner_lr)
                    # prevents the model from training for more on some tasks than others
                    if epoch_count == n_epochs:
                        logger.info(
                            f"Ending training as {n_epochs} epochs have been completed."
                        )
                        return
                    task_iterator = iter(train_dataloader)
                    X, y = next(task_iterator)
                y = metalearning_binary_target_changer(y)
                X, y = X.to(self.device), y.to(self.device)

                # copy model and setup inner_optimizer
                copied_model = deepcopy(global_model)
                copied_model.train()
                inner_optimizer = SGD(
                    copied_model.parameters(),
                    lr=self.inner_lr,  # betas=self.betas
                )  # TODO hyper-params
                lr = self.inner_lr
                for k in range(self.train_n_gradient_steps):
                    outputs = copied_model(X)
                    loss = self.loss_fn(outputs.squeeze(), y)
                    loss.backward()
                    inner_optimizer.step()
                    inner_optimizer.zero_grad()
                    lr /= self.inner_rl_reduction_factor
                    self._set_learning_rate(inner_optimizer, lr)
                    
                
                # Get metrics after last step
                with no_grad():
                    outputs = copied_model(X)
                    evaluation_error = self.loss_fn(outputs, y)
                    meta_train_error += evaluation_error.item()
                    outputs_all.append(outputs)
                    targets_all.append(y)

                # Compute the pseudo-gradient: difference between the global model and the adapted model.
                # According to the paper, one may define the meta-gradient as (global - adapted)/outer_lr.
                with no_grad():
                    for (name, global_param), (_, copied_param) in zip(
                        global_model.named_parameters(), copied_model.named_parameters()
                    ):
                        # Accumulate the difference over tasks.
                        meta_gradients[name] += (
                            global_param.detach() - copied_param.detach()
                        )

            print(
                f"Meta-update iteration {i+1}/{n_iters} complete: processed {n_parallel_tasks} parallel tasks."
            )

            # Calculate and log metric after for the current original model
            meta_train_error /= n_parallel_tasks
            big_preds = torch_cat(outputs_all, dim=0)
            big_targets = torch_cat(targets_all, dim=0)
            final_scores = compute_metrics(big_preds, big_targets)
            wandb.log(
                {
                    "train/loss": meta_train_error,
                    "train/accuracy": final_scores["accuracy"],
                    "train/f1": final_scores["f1"],
                    "train/precision": final_scores["precision"],
                    "train/recall": final_scores["recall"],
                    "train/roc_auc": final_scores["roc_auc"],
                    "epoch": epoch_count,
                    "iteration": i,
                },
            )

            # Average the accumulated meta gradients and scale by inner_lr.
            for name, param in global_model.named_parameters():
                meta_grad = meta_gradients[name] / n_parallel_tasks
                meta_grad = meta_grad / self.inner_lr
                param.grad = (
                    meta_grad.clone()
                )  # Assign the pseudo-gradient for the meta optimizer
                # print(f"params: {param.grad}")
            # Perform the meta-update using the outer optimizer.
            outer_optimizer.step()
            outer_optimizer.zero_grad()
