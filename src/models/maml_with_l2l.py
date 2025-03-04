from loguru import logger
from torch import cat as torch_cat
from torch import device as torch_device
from torch import nn, no_grad
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.models.maml_helpers_l2l as maml_helpers_l2l
import wandb
from src.data.helper_functions import metalearning_binary_target_changer
from src.models.helper_functions import batch_tasks, set_learning_rate
from src.scoring.metalearning_scoring_fn import compute_metrics


class MAML:
    def __init__(
        self,
        model: nn.Module,
        *,
        train_n_gradient_steps: int,
        eval_n_gradient_steps: int,
        device: torch_device,
        inner_lr_range: tuple[float, float],
        inner_lr_reduction_factor: int = 1.5,
        outer_lr_range: tuple[float, float],
        train_k_shot: int,
        eval_k_shot: int = None,
        loss_fn: nn.Module = None,
    ):
        model.to(device)

        self.model = model
        self.train_n_gradient_steps = train_n_gradient_steps
        self.eval_n_gradient_steps = eval_n_gradient_steps
        self.loss_fn = loss_fn or BCEWithLogitsLoss()  # TODO hypere_params?
        self.device = device
        self.inner_lr_range = inner_lr_range
        self.inner_lr = max(inner_lr_range)  # alpha from paper
        self.outer_lr_range = outer_lr_range
        self.outer_lr = max(outer_lr_range)  # beta from paper
        self.train_k_shot = train_k_shot
        self.eval_k_shot = eval_k_shot or train_k_shot
        self.inner_lr_reduction_factor = inner_lr_reduction_factor

        self.maml = maml_helpers_l2l.MAML(self.model, lr=self.inner_lr)
        self.outer_optimizer = None
        self.current_epoch = 0

    def initialize_optimizer(self):
        """Initialize or reset the optimizer with current learning rate"""
        self.outer_optimizer = SGD(self.maml.parameters(), self.outer_lr)
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

        self.outer_optimizer.zero_grad()

        # Process each task in the batch
        for task_idx in range(n_parallel_tasks):
            try:
                X, y = batch[task_idx]
            except IndexError:
                # Handle case where batch doesn't have enough tasks
                continue

            # Prep task data
            y = metalearning_binary_target_changer(y)
            X_support = X[: self.train_k_shot * 2, :].to(self.device)
            y_support = y[: self.train_k_shot * 2].to(self.device)
            X_query = X[self.train_k_shot * 2 :, :].to(self.device)
            y_query = y[self.train_k_shot * 2 :].to(self.device)

            # Clone model and adapt to task
            learner = self.maml.clone()
            learner = maml_helpers_l2l.fast_adapt(
                X_support,
                y_support,
                learner,
                self.loss_fn,
                self.train_n_gradient_steps,
                initial_lr=max(self.inner_lr_range),
                inner_rl_reduction_factor=self.inner_lr_reduction_factor,
            )

            # Make predictions and compute loss
            predictions = learner(X_query).squeeze()
            evaluation_error = self.loss_fn(predictions, y_query)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()

            predictions_all.append(predictions.detach().cpu())
            targets_all.append(y_query.detach().cpu())

        # Update model if there were tasks in the batch
        if len(predictions_all) > 0:
            # Scale gradients by number of tasks
            for p in self.maml.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / len(predictions_all))

            self.outer_optimizer.step()

            # Compute metrics
            meta_train_error /= len(predictions_all)
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
        """Evaluate the model on a batch of tasks"""
        meta_test_error = 0.0
        predictions_all = []
        targets_all = []

        # Process each task in the batch
        for X, y in batch:
            # Prep data
            X, y = X.to(self.device), y.to(self.device)
            X_support = X[: self.eval_k_shot * 2, :]
            y_support = y[: self.eval_k_shot * 2]
            X_query = X[self.eval_k_shot * 2 :, :]
            y_query = y[self.eval_k_shot * 2 :]

            # Clone and adapt model
            learner = self.maml.clone()
            learner = maml_helpers_l2l.fast_adapt(
                X_support,
                y_support,
                learner,
                self.loss_fn,
                self.eval_n_gradient_steps,  # Note: using eval steps here
                initial_lr=max(self.inner_lr_range),
                inner_rl_reduction_factor=self.inner_lr_reduction_factor,
            )

            # Evaluate
            learner.eval()
            with no_grad():
                predictions = learner(X_query).squeeze()
                evaluation_error = self.loss_fn(predictions, y_query)
                meta_test_error += evaluation_error.item()

                predictions_all.append(predictions.detach().cpu())
                targets_all.append(y_query.detach().cpu())

        # Compute metrics
        if len(predictions_all) > 0:
            meta_test_error /= len(batch)
            big_preds = torch_cat(predictions_all, dim=0)
            big_targets = torch_cat(targets_all, dim=0)
            metrics = compute_metrics(big_preds, big_targets)

            return {
                "loss": meta_test_error,
                **metrics,
                "predictions": big_preds,
                "targets": big_targets,
            }

        return None

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
    ):
        """Full training loop with optional early stopping"""
        self.model.train()
        self.initialize_optimizer()
        score_name_prefix = score_name_prefix + "." if score_name_prefix else ""

        best_metric_value = (
            float("inf") if "loss" in early_stopping_metric else -float("inf")
        )
        patience_counter = 0

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            if epoch % 10 == 0:
                pbar.set_description(f"Epoch {epoch+1}/{n_epochs}")
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
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break

            self.current_epoch = epoch
            self.update_learning_rates(epoch, n_epochs)

            # Training phase
            # epoch_train_metrics = {}
            # batch_count = 0

            batches = batch_tasks(
                train_dataloader, n_parallel_tasks
            )  # TODO batch size is wrong right?
            for i, batch in enumerate(
                tqdm(batches, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
            ):
                result = self.train_step(batch, n_parallel_tasks)
                # getting results like this, gets result of different model every iteration. So better to do after a whole epoch
                # if result:
                #     batch_count += 1
                #     for k in result:
                #         if k != "predictions" and k != "targets":
                #             epoch_train_metrics[k] = epoch_train_metrics.get(k, 0) + result[k]

            # # Average metrics
            # for k in epoch_train_metrics:
            #     epoch_train_metrics[k] /= max(1, batch_count)

            # Log training metrics
            # if log_metrics:
            #     train_log = {f"{score_name_prefix}train/{k}": v for k, v in epoch_train_metrics.items()}
            #     train_log["epoch"] = epoch
            #     wandb.log(train_log)

        train_results = self.evaluate(
            train_dataloader, f"{score_name_prefix}train", n_epochs, log_metrics
        )

        # Validation phase
        val_result = self.evaluate(
            eval_dataloader, f"{score_name_prefix}{val_or_test}", n_epochs, log_metrics=log_metrics
        )

        return train_results, val_result

    def evaluate(
        self,
        dataloader: DataLoader,
        score_name_prefix: str,
        epoch: int = None,
        log_metrics: bool = True,
        log_step: int = None,
    ):
        """Evaluate the model on the entire validation dataset"""
        self.model.eval()

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
