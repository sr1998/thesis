# This code has been adapted from the learn2learn library.
# The original code can be found at: https://github.com/learnables/learn2learn/blob/master/examples/vision/reptile_miniimagenet.py


from copy import deepcopy

from loguru import logger
from torch import cat as torch_cat
from torch import device as torch_device
from torch import nn, no_grad, zeros_like
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader

import src.models.reptile_helpers_l2l as reptile_helpers_l2l
import wandb
from src.helper_function import metalearning_binary_target_changer, set_learning_rate
from src.scoring.metalearning_scoring_fn import compute_metrics


class Reptile:  # Assumes binary classifier for now
    def __init__(
        self,
        model: nn.Module,
        *,
        train_n_gradient_steps: int,
        eval_n_gradient_steps: int,
        device: torch_device,
        inner_lr_range: tuple[float, float],
        outer_lr_range: tuple[float, float],
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
        self.inner_lr_range = inner_lr_range
        self.inner_lr = max(inner_lr_range)
        self.inner_rl_reduction_factor = inner_rl_reduction_factor
        self.outer_lr_range = outer_lr_range
        self.outer_lr = max(outer_lr_range)
        self.betas = betas or (0.0, 0.999)
        self.k_shot = k_shot

    def evaluate(self, dataloader: DataLoader, epoch: int):
        """Evaluate the current model on the full dataset.

        This differs from training, as it does the inner loop on a support set but evaluates on a query set,
        that is usually the rest of the data for the classes in the task.
        """
        meta_test_error = 0.0
        outputs_all = []
        targets_all = []

        learner = deepcopy(self.model)
        inner_optimizer = Adam(
            learner.parameters(), lr=self.inner_lr, betas=self.betas
        )
        inner_optimizer.load_state_dict(self.inner_optimizer_state)
        for X, y in dataloader:
            # get support/query data
            X, y = X.to(self.device), y.to(self.device)
            X_support = X[: self.k_shot * 2, :]
            y_support = y[: self.k_shot * 2]
            X_query = X[self.k_shot * 2 :, :]
            y_query = y[self.k_shot * 2 :]
            learner = reptile_helpers_l2l.fast_adapt(
                X_support,
                y_support,
                learner,
                self.loss_fn,
                inner_optimizer,
                self.train_n_gradient_steps,
                self.inner_lr,
                self.inner_rl_reduction_factor,
            )
            learner.eval()
            with no_grad():
                outputs = learner(X_query).squeeze()
                eval_error = self.loss_fn(outputs, y_query)
                meta_test_error += eval_error.item()

                outputs_all.append(outputs.detach().cpu())
                targets_all.append(y_query.detach().cpu())

        # # Compute final metrics
        meta_test_error /= len(dataloader)
        big_preds = torch_cat(outputs_all, dim=0)
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

        # we want each epoch to go through the whole dataset
        n_iters = n_epochs * (len(train_dataloader) // n_parallel_tasks + 1)
        logger.info(
            f"Starting training with {n_iters} iterations and {n_parallel_tasks} parallel tasks."
        )
        epoch_count = 0

        self.model.train()
        outer_optimizer = SGD(self.model.parameters(), self.outer_lr)
        inner_optimizer = Adam(
            self.model.parameters(), lr=self.inner_lr, betas=self.betas
        )
        self.inner_optimizer_state = inner_optimizer.state_dict()

        # new_outer_lr = self.outer_lr * (1.0 - (epoch_count / float(n_epochs)) ** 0.5)
        # self._set_learning_rate(new_outer_lr)
        if evaluate_train:
            self.evaluate(val_dataloader, epoch_count)
        task_iterator = iter(train_dataloader)

        for i in range(n_iters):
            outer_optimizer.zero_grad()
            meta_train_error = 0.0
            outputs_all = []
            targets_all = []

            # zero-grad the parameters
            for p in self.model.parameters():
                p.grad = zeros_like(p.data)

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
                    # Update outer_lr
                    self.outer_lr = min(self.outer_lr_range) * (
                        epoch_count / n_epochs
                    ) + max(self.outer_lr_range) * (1 - epoch_count / n_epochs)
                    set_learning_rate(outer_optimizer, self.outer_lr)

                    logger.info(
                        f"Epoch {epoch_count} complete at iteration {i+1}/{n_iters} with {n_parallel_tasks} parallel tasks and {len(train_dataloader)} total tasks. Reinitializing DataLoader for next epoch."
                    )

                    if evaluate_train:
                        self.evaluate(val_dataloader, epoch_count)
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

                learner = deepcopy(self.model)
                inner_optimizer = Adam(
                    learner.parameters(), lr=self.inner_lr, betas=self.betas
                )
                inner_optimizer.load_state_dict(self.inner_optimizer_state)
                learner = reptile_helpers_l2l.fast_adapt(
                    X,
                    y,
                    learner,
                    self.loss_fn,
                    inner_optimizer,
                    self.train_n_gradient_steps,
                    self.inner_lr,
                    self.inner_rl_reduction_factor,
                )
                outputs = learner(X).squeeze()
                train_error = self.loss_fn(outputs, y)

                self.inner_optimizer_state = inner_optimizer.state_dict()
                for p, l in zip(self.model.parameters(), learner.parameters()):
                    p.grad.data.add(l.data, alpha=-1.0)

                meta_train_error += train_error.item()
                outputs_all.append(outputs.detach().cpu())
                targets_all.append(y.detach().cpu())
                

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

            # Average the accumulated gradients and optimize
            for p in self.model.parameters():
                p.grad.data.mul_(1.0 / n_parallel_tasks).add_(p.data)
            outer_optimizer.step()
