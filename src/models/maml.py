from copy import deepcopy

import higher
from loguru import logger
from torch import Tensor, nn, no_grad, tensor, zeros_like
from torch import device as torch_device
from torch.autograd import grad
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader

import wandb
from src.helper_function import metalearning_binary_target_changer

import src.models.maml_learn2learn as l2l


class MAML:
    def __init__(
        self,
        model: nn.Module,
        *,
        train_n_gradient_steps: int,
        eval_n_gradient_steps: int,
        device: torch_device,
        meta_optimizer: Optimizer,
        inner_lr_range: tuple[float, float],
        outer_lr_range: tuple[float, float],
        k_shot: int,
        loss_fn: nn.Module = None,
    ):
        model.to(device)

        self.model = model
        self.train_n_gradient_steps = train_n_gradient_steps
        self.eval_n_gradient_steps = eval_n_gradient_steps
        self.loss_fn = loss_fn or BCEWithLogitsLoss()  # TODO hypere_params?
        self.device = device
        self.outer_optimizer = meta_optimizer
        self.inner_lr_range = inner_lr_range
        self.inner_lr = max(inner_lr_range)  # alpha from paper
        self.outer_lr_range = outer_lr_range
        self.outer_lr = max(outer_lr_range)  # beta from paper
        self.k_shot = k_shot

        self.maml = l2l.MAML(self.model, lr=self.inner_lr)

    def _set_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

   

    def inner_loop(self, X_support: Tensor, y_support: Tensor, train: bool = True):
        inner_optimizer = SGD(self.model.parameters(), lr=self.inner_lr)
        
        # Create a functional (fast) model that supports differentiable updates.
        # with higher.innerloop_ctx(
        #     self.model,
        #     inner_optimizer,
        #     copy_initial_weights=True,
        #     track_higher_grads=train,
        # ) as (fmodel, diffopt):
        #     steps = self.train_n_gradient_steps if train else self.eval_n_gradient_steps
        #     for _ in range(steps):
        #         # Forward pass on support data
        #         support_preds = fmodel(X_support).squeeze()
        #         support_loss = self.loss_fn(support_preds, y_support)
        #         # Perform an inner-loop update (this update is differentiable)
        #         diffopt.step(support_loss)
        #     return fmodel

        # Deepcopy the model
        task_model = type(self.model)(X_support.shape[1]).to(self.device)
        task_model.load_state_dict(self.model.state_dict())
        task_model.train()
        for i in range(
            self.train_n_gradient_steps if train else self.eval_n_gradient_steps
        ):
            # Support loss
            support_loss = self.loss_fn(task_model(X_support).squeeze(), y_support)

            # Get gradients w.r.t. task_model
            grads = grad(support_loss, task_model.parameters(), create_graph=train)
            # print("inner lr:", self.inner_lr)
            # print("grads", grads, sep="\n")

            # with no_grad():   # TODO does this make a difference?
            for param, g in zip(task_model.parameters(), grads):
                param -= self.inner_lr * g

        return task_model

    def fit(
        self,
        *,
        train_dataloader: DataLoader,
        n_epochs: int,
        n_parallel_tasks: int,
        evaluate_train: bool = False,
        val_dataloader: DataLoader = None,
    ):
        # we want each epoch to go through the whole dataset
        n_iters = n_epochs * (len(train_dataloader) // n_parallel_tasks + 1)

        task_iterator = iter(train_dataloader)

        epoch_count = 0
        self._set_learning_rate(self.outer_optimizer, self.outer_lr)
        if evaluate_train:
            self.evaluate(val_dataloader, epoch_count)

        self.model.train()

        opt = SGD(self.maml.parameters(), self.outer_lr)

        for i in range(n_iters):
            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            opt.zero_grad()
            # Sum of losses for meta-learning updates
            # meta_loss = tensor([0.0], requires_grad=True).to(self.device)

            # Inner loop updates due to each task
            for task in range(n_parallel_tasks):
                # Get a task
                try:
                    X, y = next(task_iterator)
                except StopIteration:
                    epoch_count += 1
                    self.outer_lr = min(self.outer_lr_range) * (epoch_count / n_epochs) + max(self.outer_lr_range) * (1 - epoch_count / n_epochs)
                    self._set_learning_rate(self.outer_optimizer, self.outer_lr)
                    self.inner_lr = min(self.inner_lr_range) * (epoch_count / n_epochs) + max(self.inner_lr_range) * (1 - epoch_count / n_epochs)
                    self.maml.lr = self.inner_lr
                    logger.info(
                        f"Epoch {epoch_count} complete at iteration {i+1}/{n_iters} with {n_parallel_tasks} parallel tasks and {len(train_dataloader)} total tasks. Reinitializing DataLoader for next epoch."
                    )
                    if evaluate_train:
                        self.evaluate(val_dataloader, epoch_count)

                    if epoch_count == n_epochs:
                        logger.info(
                            f"Ending training as {n_epochs} epochs have been completed."
                        )
                        return
                    task_iterator = iter(train_dataloader)
                    X, y = next(task_iterator)
                # Prep task data
                y = metalearning_binary_target_changer(y)
                X_support = X[: self.k_shot * 2, :].to(self.device)
                y_support = y[: self.k_shot * 2].to(self.device)
                X_query = X[self.k_shot * 2 :, :].to(self.device)
                y_query = y[self.k_shot * 2 :].to(self.device)

                learner = self.maml.clone()
                evaluation_error, evaluation_accuracy = l2l.fast_adapt(X_support, y_support, X_query, y_query, learner, self.loss_fn, self.train_n_gradient_steps)
                evaluation_error.backward()

                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()
                
            meta_train_error /= n_parallel_tasks
            meta_train_accuracy /= n_parallel_tasks

            #     task_model = self.inner_loop(X_support, y_support)
            #     # Query loss with updated fast_model
            #     query_preds = task_model(X_query).squeeze()
            #     query_loss = self.loss_fn(query_preds, y_query)

            #     meta_loss += query_loss

            # Average the accumulated gradients and optimize
            for p in self.maml.parameters():
                p.grad.data.mul_(1.0 / n_parallel_tasks)
            opt.step()

            # # Update of the original model (outer loop)
            # print(self.model.state_dict())
            # meta_loss /= n_parallel_tasks
            # meta_loss.backward()
            # self.outer_optimizer.step()
            # self.outer_optimizer.zero_grad()
            # print(self.model.state_dict())

    def evaluate(self, dataloader: DataLoader, epoch: int):
        # total_loss = 0.0
        # total_correct = 0
        # total_samples = 0

        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        for X, y in dataloader:
            # get support/query data
            X, y = X.to(self.device), y.to(self.device)
            X_support = X[: self.k_shot * 2, :]
            y_support = y[: self.k_shot * 2]
            X_query = X[self.k_shot * 2 :, :]
            y_query = y[self.k_shot * 2 :]

            # Get task model with inner loop updates
        #     task_model = self.inner_loop(X_support, y_support, train=True)
        #     task_model.eval()
        #     with no_grad():
        #         # Get predictions with this model
        #         query_preds = task_model(X_query).squeeze()
        #         # Calculate loss
        #         query_loss = self.loss_fn(query_preds, y_query)
        #         total_loss += query_loss
        #         # Calculate accuracy vars
        #         y_preds = (query_preds > 0).float().squeeze()
        #         total_correct += (y_preds == y_query).sum().item()
        #         total_samples += y_query.shape[0]

            learner = self.maml.clone()
            evaluation_error, evaluation_accuracy = l2l.fast_adapt(X_support, y_support, X_query, y_query, learner, self.loss_fn, self.train_n_gradient_steps)

            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        # # Compute final metrics
        # avg_loss = total_loss / len(
        #     dataloader
        # )  # TODO check whether len is 1 indeed for our case
        # avg_accuracy = total_correct / total_samples
        meta_test_error /= len(dataloader)
        meta_test_accuracy /= len(dataloader)
        logger.info(f"Evaluation after epoch {epoch}: Accuracy = {meta_test_accuracy:.2f}")
        logger.info(f"Evaluation after epoch {epoch}: Loss = {meta_test_error:.2f}")


 # def learn_from_tasks(self, x_spt, y_spt, x_qry, y_qry):
    #     """

    #     :param x_spt:   [n_tasks, supportsz, n_features]
    #     :param y_spt:   [n_tasks, supportsz]
    #     :param x_qry:   [n_tasks, querysz, n_features]
    #     :param y_qry:   [n_tasks, querysz]
    #     :return:
    #     """
    #     n_tasks, supportsz, n_features = x_spt.size()
    #     querysz = x_qry.size(1)

    #     losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
    #     corrects = [0 for _ in range(self.update_step + 1)]

    #     for i in range(n_tasks):

    #         # 1. run the i-th task and compute loss for k=0
    #         logits = self.model(x_spt[i], vars=None, bn_training=True)
    #         loss = F.cross_entropy(logits, y_spt[i])
    #         grad = torch.autograd.grad(loss, self.net.parameters())
    #         fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

    #         # this is the loss and accuracy before first update
    #         with torch.no_grad():
    #             # [supportsz, nway]
    #             logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
    #             loss_q = F.cross_entropy(logits_q, y_qry[i])
    #             losses_q[0] += loss_q

    #             pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
    #             correct = torch.eq(pred_q, y_qry[i]).sum().item()
    #             corrects[0] = corrects[0] + correct

    #         # this is the loss and accuracy after the first update
    #         with torch.no_grad():
    #             # [supportsz, nway]
    #             logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
    #             loss_q = F.cross_entropy(logits_q, y_qry[i])
    #             losses_q[1] += loss_q
    #             # [supportsz]
    #             pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
    #             correct = torch.eq(pred_q, y_qry[i]).sum().item()
    #             corrects[1] = corrects[1] + correct

    #         for k in range(1, self.update_step):
    #             # 1. run the i-th task and compute loss for k=1~K-1
    #             logits = self.net(x_spt[i], fast_weights, bn_training=True)
    #             loss = F.cross_entropy(logits, y_spt[i])
    #             # 2. compute grad on theta_pi
    #             grad = torch.autograd.grad(loss, fast_weights)
    #             # 3. theta_pi = theta_pi - train_lr * grad
    #             fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

    #             logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
    #             # loss_q will be overwritten and just keep the loss_q on last update step.
    #             loss_q = F.cross_entropy(logits_q, y_qry[i])
    #             losses_q[k + 1] += loss_q

    #             with torch.no_grad():
    #                 pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
    #                 correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
    #                 corrects[k + 1] = corrects[k + 1] + correct

    #     # end of all tasks
    #     # sum over all losses on query set across all tasks
    #     loss_q = losses_q[-1] / n_tasks

    #     # optimize theta parameters
    #     self.meta_optim.zero_grad()
    #     loss_q.backward()
    #     # print('meta update')
    #     # for p in self.net.parameters()[:5]:
    #     # 	print(torch.norm(p).item())
    #     self.meta_optim.step()

    #     accs = np.array(corrects) / (querysz * n_tasks)

    #     return accs