# This code has been adapted from the learn2learn library.
# The original code can be found at: https://github.com/learnables/learn2learn/blob/master/examples/vision/reptile_miniimagenet.py


from copy import deepcopy

from loguru import logger
from torch import cat as torch_cat
from torch import device as torch_device
from torch import nn, no_grad, zeros_like
from torch import save as torch_save
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader

import src.models.reptile_helpers_l2l as reptile_helpers_l2l
import wandb
from src.data.helper_functions import metalearning_binary_target_changer
from src.models.helper_functions import batch_tasks, set_learning_rate
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
        inner_lr_reduction_factor: float,
        train_k_shot: int,
        eval_k_shot: int = None,
        betas: tuple[float, float] = None,
        loss_fn: nn.Module = None,
        weight_decay: float = 0.0,
    ):
        model.to(device)

        self.model = model
        self.train_n_gradient_steps = train_n_gradient_steps
        self.eval_n_gradient_steps = eval_n_gradient_steps
        self.loss_fn = loss_fn or BCEWithLogitsLoss()
        self.device = device
        self.inner_lr_range = inner_lr_range
        self.inner_lr = max(inner_lr_range)
        self.inner_lr_reduction_factor = inner_lr_reduction_factor
        self.outer_lr_range = outer_lr_range
        self.outer_lr = max(outer_lr_range)
        self.betas = betas or (0.0, 0.999)
        self.train_k_shot = train_k_shot
        self.eval_k_shot = eval_k_shot or train_k_shot
        self.weight_decay = weight_decay

        self.outer_optimizer = None
        self.current_epoch = 0

        # Initialize inner optimizer state that can be reused across tasks
        inner_optimizer = Adam(
            self.model.parameters(), lr=self.inner_lr, betas=self.betas
        )
        self.inner_optimizer_state = inner_optimizer.state_dict()

    def initialize_optimizer(self):
        """Initialize or reset the optimizer with current learning rate"""
        self.outer_optimizer = SGD(
            self.model.parameters(), self.outer_lr, weight_decay=self.weight_decay
        )
        return self.outer_optimizer

    def update_learning_rates(self, epoch, total_epochs):
        """Update learning rates based on current epoch"""
        self.outer_lr = min(self.outer_lr_range) * (epoch / total_epochs) + max(
            self.outer_lr_range
        ) * (1 - epoch / total_epochs)
        if self.outer_optimizer:
            set_learning_rate(self.outer_optimizer, self.outer_lr)
        return self.outer_lr

    def train_step(self, batch, n_parallel_tasks=1):
        """Perform a single training step on a batch of tasks"""
        if self.outer_optimizer is None:
            self.initialize_optimizer()

        meta_train_error = 0.0
        predictions_all = []
        targets_all = []

        # Track task-specific adapted parameters
        task_adapted_params = []
        num_tasks_processed = 0

        self.model.train()

        # Process each task in the batch
        for task_idx in range(n_parallel_tasks):
            try:
                X, y = batch[task_idx]
            except IndexError:
                # Handle case where batch doesn't have enough tasks
                continue

            # Prep task data
            y = metalearning_binary_target_changer(y)
            X = X.to(self.device)
            y = y.to(self.device)

            # Clone model and adapt to task
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
                initial_lr=self.inner_lr,
                inner_lr_reduction_factor=self.inner_lr_reduction_factor,
            )

            # Store the adapted parameters for this task
            task_adapted_params.append([p.data.clone() for p in learner.parameters()])
            num_tasks_processed += 1

            # For tracking metrics, evaluate on the same data
            learner.eval()
            with no_grad():
                predictions = learner(X).squeeze()
                evaluation_error = self.loss_fn(predictions, y)
                meta_train_error += evaluation_error.item()

                predictions_all.append(predictions.detach().cpu())
                targets_all.append(y.detach().cpu())

        # Update model if there were tasks in the batch
        if num_tasks_processed > 0:
            # True batched Reptile update as per the paper (equation 5)
            # φ ← φ + (ε/n) * Σ(φ̃ᵢ - φ)
            self.outer_optimizer.zero_grad()

            # Compute the average of differences between adapted parameters and original parameters
            for i, param in enumerate(self.model.parameters()):
                # Initialize gradient
                param_diff = zeros_like(param.data)

                # Sum up all the differences (φ̃ᵢ - φ) for each task
                for task_params in task_adapted_params:
                    param_diff += task_params[i] - param.data

                # Average the differences
                param_diff /= num_tasks_processed

                # Set the gradient to be the parameter difference
                # (This works because the optimizer will do: param = param - lr * grad)
                # where we want: param = param + lr * avg_(adapted - param)
                param.grad = (
                    -param_diff
                )  # Negative because optimizers do gradient descent

            # Apply the update
            self.outer_optimizer.step()

            # Update the inner optimizer state to match the new model parameters
            # This ensures that future task adaptations start with an optimizer
            # state that's consistent with the updated parameters
            temp_inner_optimizer = Adam(
                self.model.parameters(), lr=self.inner_lr, betas=self.betas
            )
            self.inner_optimizer_state = temp_inner_optimizer.state_dict()

            # Compute metrics
            meta_train_error /= num_tasks_processed
            big_preds = torch_cat(predictions_all, dim=0)
            big_targets = torch_cat(targets_all, dim=0)
            metrics = compute_metrics(big_preds, big_targets)

            return {
                "loss": meta_train_error,
                **metrics,
                "predictions": big_preds,
                "targets": big_targets,
            }

        return None

    def evaluate_step(self, batch):
        """Evaluate the current model on the full dataset.

        This differs from training, as it does the inner loop on a support set but evaluates on a query set,
        that is usually the rest of the data for the classes in the task.
        """
        meta_test_error = 0.0
        outputs_all = []
        targets_all = []

        # Backup the training optimizer state to avoid information leakage
        # as recommended in the Reptile paper
        original_optimizer_state = deepcopy(self.inner_optimizer_state)

        self.model.train()
        # Process each task in the batch
        for X, y in batch:
            # Prep data
            X, y = X.to(self.device), y.to(self.device)
            X_support = X[: self.eval_k_shot * 2, :]
            y_support = y[: self.eval_k_shot * 2]
            X_query = X[self.eval_k_shot * 2 :, :]
            y_query = y[self.eval_k_shot * 2 :]

            # Clone and adapt model
            learner = deepcopy(self.model)

            # Create a fresh optimizer for evaluation to prevent information leakage
            # Set β1 = 0 for Adam as recommended in the paper
            inner_optimizer = Adam(
                learner.parameters(), lr=self.inner_lr, betas=self.betas
            )
            # Note: Not loading the training optimizer state here - fresh optimizer

            # Adapt the model
            learner = reptile_helpers_l2l.fast_adapt(
                X_support,
                y_support,
                learner,
                self.loss_fn,
                inner_optimizer,
                self.eval_n_gradient_steps,  # Note: using eval steps here
                self.inner_lr,
                self.inner_lr_reduction_factor,
            )

            # Evaluate
            learner.eval()
            with no_grad():
                predictions = learner(X_query).squeeze()
                evaluation_error = self.loss_fn(predictions, y_query)
                meta_test_error += evaluation_error.item()

                outputs_all.append(predictions.detach().cpu())
                targets_all.append(y_query.detach().cpu())

        # Restore the original optimizer state after evaluation
        self.inner_optimizer_state = original_optimizer_state

        # Compute metrics
        if len(outputs_all) > 0:
            meta_test_error /= len(batch)
            big_preds = torch_cat(outputs_all, dim=0)
            big_targets = torch_cat(targets_all, dim=0)
            metrics = compute_metrics(big_preds, big_targets)

            return {
                "loss": meta_test_error,
                **metrics,
                "predictions": big_preds,
                "targets": big_targets,
            }

        return None

    def evaluate(
        self,
        dataloader: DataLoader,
        score_name_prefix: str,
        epoch: int = None,
        log_metrics: bool = True,
        log_step: int = None,
    ):
        """Evaluate the model on the entire validation dataset"""

        all_batches = list(dataloader)
        results = self.evaluate_step(all_batches)

        if log_metrics and results:
            val_log = {
                f"{score_name_prefix}/loss": results["loss"],
                f"{score_name_prefix}/accuracy": results["accuracy"],
                f"{score_name_prefix}/f1": results["f1"],
                f"{score_name_prefix}/precision": results["precision"],
                f"{score_name_prefix}/recall": results["recall"],
                f"{score_name_prefix}/roc_auc": results["roc_auc"],
                "epoch": epoch,
                "log_step": log_step,
            }
            wandb.log(val_log)

            logger.info(f"Evaluation after epoch {epoch}: Loss = {results['loss']:.2f}")
            logger.info(
                f"Accuracy = {results['accuracy']:.2f}, "
                + f"F1 = {results['f1']:.2f}, "
                + f"Precision = {results['precision']:.2f}, "
                + f"Recall = {results['recall']:.2f}, "
                + f"ROC-AUC = {results['roc_auc']:.2f}"
            )

        self.model.train()
        return results

    def fit(
        self,
        *,
        train_dataloader: DataLoader,
        n_epochs: int,
        n_parallel_tasks: int,
        eval_dataloader: DataLoader = None,
        val_or_test: str = "val",
        early_stopping_patience: int = None,
        early_stopping_metric: str = "loss",
        log_metrics: bool = True,
        score_name_prefix: str = None,
        save_best_model_path: str = None,
    ):
        self.model.train()
        self.initialize_optimizer()
        score_name_prefix = score_name_prefix + "." if score_name_prefix else ""

        best_metric_value = (
            float("inf") if "loss" in early_stopping_metric else -float("inf")
        )
        patience_counter = 0

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}")

            # log every 10 epochs, overwriting previous log
            if log_metrics and epoch % 10 == 0:
                train_results = self.evaluate(
                    train_dataloader, f"{score_name_prefix}train", epoch, log_metrics
                )

            # Validation phase
            if eval_dataloader:
                val_result = self.evaluate(
                    eval_dataloader,
                    f"{score_name_prefix}{val_or_test}",
                    epoch,
                    log_metrics=True if epoch % 10 == 0 and log_metrics else False,
                )

                # Early stopping check
                if early_stopping_patience:
                    current_metric = val_result[early_stopping_metric]

                    improved = (
                        early_stopping_metric == "loss"
                        and current_metric < best_metric_value
                    ) or (
                        early_stopping_metric != "loss"
                        and current_metric > best_metric_value
                    )

                    if improved:
                        best_metric_value = current_metric
                        patience_counter = 0

                        # Save the best model
                        if save_best_model_path:
                            torch_save(self, save_best_model_path)
                            logger.info(
                                f"Saved new best model with {early_stopping_metric} = {current_metric:.4f}"
                            )
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break

            self.current_epoch = epoch
            self.update_learning_rates(epoch, n_epochs)

            # Training phase
            batches = batch_tasks(train_dataloader, n_parallel_tasks)
            for batch in batches:
                self.train_step(batch, n_parallel_tasks)

        # Final evaluation
        train_results = self.evaluate(
            train_dataloader, f"{score_name_prefix}train", n_epochs, log_metrics
        )

        # Validation phase
        val_result = None
        if eval_dataloader:
            val_result = self.evaluate(
                eval_dataloader,
                f"{score_name_prefix}{val_or_test}",
                n_epochs,
                log_metrics=log_metrics,
            )

        return train_results, val_result
