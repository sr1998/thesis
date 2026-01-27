from loguru import logger
import torch
from torch import cat as torch_cat
from torch import device as torch_device
from torch import nn, no_grad
from torch import save as torch_save
from torch import load as torch_load
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
        device: torch_device,
        train_k_shot: int,
        eval_k_shot: int = None,
        # New parameters to match ProtoNet
        starting_lr: float = 0.01,
        scheduler_step: int = 50,
        scheduler_gamma: float = 0.5,
        class_weights_loss_fn = (.5, .5),
        evaluate_every: int = 10,
        # Original parameters (renamed or with defaults)
        train_n_gradient_steps: int = 5,
        eval_n_gradient_steps: int = 5,
        inner_lr_range: tuple[float, float] = (0.001, 0.1),
        inner_lr_reduction_factor: int = 1.5,
        outer_lr_range: tuple[float, float] = None,  # Derive from starting_lr if None
        loss_fn: nn.Module = None,
        weight_decay: float = 0.0,
    ):
        model.to(device)
        
        # Store original MAML parameters
        self.model = model
        self.train_n_gradient_steps = train_n_gradient_steps
        self.eval_n_gradient_steps = eval_n_gradient_steps
        self.class_weights_loss_fn = torch.tensor(class_weights_loss_fn).to(device)
        self.loss_fn = loss_fn or BCEWithLogitsLoss(self.class_weights_loss_fn)
        self.device = device
        self.inner_lr_range = inner_lr_range
        self.inner_lr = max(inner_lr_range)
        # self.outer_lr_range = outer_lr_range or (starting_lr / 10, starting_lr)
        self.outer_lr = starting_lr  # Use starting_lr as initial outer_lr
        self.train_k_shot = train_k_shot
        self.eval_k_shot = eval_k_shot or train_k_shot
        self.inner_lr_reduction_factor = inner_lr_reduction_factor
        self.weight_decay = weight_decay
        
        # Store ProtoNet-specific parameters
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.evaluate_every = evaluate_every
        
        # Initialize MAML components
        self.maml = maml_helpers_l2l.MAML(self.model, lr=self.inner_lr)
        self.maml.to(self.device)
        
        # Change to Adam optimizer to match ProtoNet
        self.outer_optimizer = torch.optim.Adam(
            self.maml.parameters(),
            lr=self.outer_lr,
            weight_decay=self.weight_decay,
        )
        
        # Add scheduler like ProtoNet
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.outer_optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma,
        )
        
        self.current_epoch = 0

    def _log_gradients(self, epoch, score_name_prefix):
        """Log gradient statistics to wandb"""
        if (epoch + 1) % 10 == 0:
            with no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # Log gradient statistics
                        gradient_log = {
                            f"{score_name_prefix}gradients/{name}_norm": param.grad.norm().item(),
                            f"{score_name_prefix}gradients/{name}_mean": param.grad.mean().item(),
                            f"{score_name_prefix}gradients/{name}_max": param.grad.max().item(),
                            f"{score_name_prefix}gradients/{name}_min": param.grad.min().item(),
                            f"{score_name_prefix}gradients/{name}_histogram": wandb.Histogram(
                                param.grad.detach().cpu().numpy().flatten()
                            ),
                            "epoch": epoch + 1,
                        }

                        # Only calculate std if there are at least 2 elements
                        if param.grad.numel() > 1:
                            gradient_log[f"{score_name_prefix}gradients/{name}_std"] = (
                                param.grad.std().item()
                            )

                        wandb.log(gradient_log)

    def initialize_optimizer(self):
        """Initialize or reset the optimizer with current learning rate"""
        self.outer_optimizer = SGD(
            self.maml.parameters(), self.outer_lr, weight_decay=self.weight_decay
        )
        return self.outer_optimizer

    # def update_learning_rates(self, epoch, total_epochs):
    #     """Update learning rates based on current epoch"""
    #     self.outer_lr = min(self.outer_lr_range) * (epoch / total_epochs) + max(
    #         self.outer_lr_range
    #     ) * (1 - epoch / total_epochs)
    #     if self.outer_optimizer:
    #         set_learning_rate(self.outer_optimizer, self.outer_lr)
    #     return self.outer_lr

    def train_step(self, batch, log_gradients, n_parallel_tasks, score_name_prefix=None):
        """Perform a single training step on a batch of tasks"""
        if self.outer_optimizer is None:
            self.initialize_optimizer()

        meta_train_error = 0.0
        predictions_all = []
        targets_all = []

        self.maml.train()
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
            X = X.to(self.device)
            y = y.to(self.device)

            X_support = X[: self.train_k_shot * 2, :]
            y_support = y[: self.train_k_shot * 2]
            X_query = X[self.train_k_shot * 2 :, :]
            y_query = y[self.train_k_shot * 2 :]

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
            evaluation_error = self.loss_fn(predictions, y_query, )
            evaluation_error.backward()
            
            if log_gradients:
                self._log_gradients(self.current_epoch, score_name_prefix or "")
            meta_train_error += evaluation_error.item()

            predictions_all.append(predictions.detach().cpu())
            targets_all.append(y_query.detach().cpu())

        # Update model if there were tasks in the batch
        if len(predictions_all) > 0:
            # Scale gradients by number of tasks
            for p in self.maml.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / len(predictions_all))

            # # clip gradients
            # nn.utils.clip_grad_norm_(self.maml.parameters(), max_norm=1.0)

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

    def evaluate_step(self, dataloader):
        """Evaluate the model on a batch of tasks"""
        meta_test_error = 0.0
        predictions_all = []
        targets_all = []

        data_iter = iter(dataloader)
        task_count = 0

        # Process each task in the batch
        for batch in batch_tasks(data_iter, 1):
            # Prep data
            X, y = batch[0] 
            X, y = X.to(self.device), y.to(self.device)
            X_support = X[: self.eval_k_shot * 2, :]
            y_support = y[: self.eval_k_shot * 2]
            X_query = X[self.eval_k_shot * 2 :, :]
            y_query = y[self.eval_k_shot * 2 :]

            # Clone and adapt model
            learner = self.maml.clone()
            learner.train()
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
                task_count += 1

        # Compute metrics
        if len(predictions_all) > 0:
            meta_test_error /= task_count
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
        eval_dataloader: DataLoader = None,
        val_dataloader: DataLoader = None,  # New parameter for compatibility
        val_or_test: str = "val",
        early_stopping_patience: int = None,
        early_stopping_metric: str = "loss",
        early_stopping_fraction: float = 0.0,  # New parameter
        log_metrics: bool = True,
        log_gradients: bool = False,
        score_name_prefix: str = None,
        save_best_model_path: str = None,
        track_best_f1: bool = True,
        load_best_model: bool = False,  # New parameter
        accumulation_steps: int = 1,  # Just for interface compatibility
        n_parallel_tasks: int = 1,  # Keep this for MAML's training logic
        **kwargs  # Accept extra kwargs for compatibility
    ):
        """Full training loop with optional early stopping - modified to match ProtoNet interface"""
        self.maml.train()
        score_name_prefix = score_name_prefix + "." if score_name_prefix else ""
        
        # Use eval_dataloader if provided, otherwise fall back to val_dataloader
        if eval_dataloader is None and val_dataloader is not None:
            eval_dataloader = val_dataloader

        best_metric_value = float("inf") if "loss" in early_stopping_metric else -float("inf")
        patience_counter = 0
        
        # Calculate early stopping batches if needed
        n_es = max(1, int(len(train_dataloader) * early_stopping_fraction))

        # Best F1 tracking
        best_f1 = -float("inf")
        best_f1_epoch = 0
        best_model_state = None
        all_f1_scores = []

        for epoch in range(n_epochs):
            # Log periodically based on evaluate_every
            if epoch % self.evaluate_every == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}")
                
                if log_metrics:
                    train_results = self.evaluate(
                        train_dataloader, f"{score_name_prefix}train", epoch, log_metrics
                    )

                # Validation phase
                if eval_dataloader  and (epoch % self.evaluate_every == 0 or epoch <= 20):
                    val_result = self.evaluate(
                        eval_dataloader,
                        f"{score_name_prefix}{val_or_test}",
                        epoch,
                        log_metrics=log_metrics,
                    )
                    all_f1_scores.append(val_result["f1"])

                    # Track best F1 score
                    if track_best_f1 and "f1" in val_result and val_result["f1"] > best_f1 and epoch > 0.1*n_epochs:
                        best_f1 = val_result["f1"]
                        best_f1_epoch = epoch
                        
                        # Save model state with best F1
                        # if save_best_model_path:
                        #     best_model_state = {
                        #         'model_state_dict': self.model.state_dict(),
                        #         'maml_state_dict': self.maml.state_dict(),
                        #         'epoch': epoch,
                        #         'f1_score': best_f1
                        #     }
                        #     torch_save(best_model_state, str(save_best_model_path) + ".best_f1")
                        #     logger.info(f"Saved new best F1 model with F1 = {best_f1:.4f} at epoch {epoch+1}")

            # Training phase - KEEP ORIGINAL MAML TRAINING APPROACH
            self.current_epoch = epoch
            
            train_iter = iter(train_dataloader)

            # Early stopping check (matches ProtoNet logic)
            if early_stopping_patience and early_stopping_fraction > 0:
                # Get a sample of tasks for early stopping
                es_batches = []
                # Get early stopping data
                with no_grad():
                    for _ in range(n_es):
                        X, y = next(train_iter)
                        es_batches.append((X.clone().detach(), y.clone().detach()))
                early_stopping_res = self.evaluate(
                    es_batches, f"{score_name_prefix}_es", epoch, log_metrics=False
                )
                current_metric = early_stopping_res[early_stopping_metric]
                        
                improved = (
                    early_stopping_metric == "loss" and current_metric < best_metric_value
                ) or (
                    early_stopping_metric != "loss" and current_metric > best_metric_value
                )

                if improved:
                    best_metric_value = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Original MAML training logic
            for batch in batch_tasks(train_iter, n_parallel_tasks):
                # Use the original train_step method for MAML
                result = self.train_step(
                    batch, 
                    log_gradients=(epoch % self.evaluate_every == 0), 
                    n_parallel_tasks=n_parallel_tasks,
                    score_name_prefix=score_name_prefix
                )
                
                # Log training metrics if requested
                if log_metrics and result and epoch % self.evaluate_every == 0:
                    wandb.log({
                        f"{score_name_prefix}train/batch_loss": result["loss"],
                        "epoch": epoch
                    })
            
            # Update learning rate with scheduler
            self.scheduler.step()

        # Final evaluation
        train_results = self.evaluate(
            train_dataloader, f"{score_name_prefix}train", n_epochs, log_metrics
        )

        val_result = None
        if eval_dataloader:
            val_result = self.evaluate(
                eval_dataloader,
                f"{score_name_prefix}{val_or_test}",
                n_epochs,
                log_metrics=log_metrics,
            )
            all_f1_scores.append(val_result["f1"])

            # Include best F1 information in results
            if track_best_f1:
                val_result["best_f1"] = best_f1
                val_result["best_f1_epoch"] = best_f1_epoch
                
                if log_metrics:
                    wandb.log({
                        f"{score_name_prefix}{val_or_test}/best_f1": best_f1,
                        f"{score_name_prefix}{val_or_test}/best_f1_epoch": best_f1_epoch,
                        "epoch": n_epochs
                    })
                    
            # Add average F1 score over last 10 epochs
            val_result["averaged_f1_score"] = sum(all_f1_scores[-10:]) / min(10, len(all_f1_scores))
            
            # Log the best F1 separately
            if log_metrics:
                import wandb
                wandb.log({
                    f"{score_name_prefix}{val_or_test}/best_f1": best_f1,
                    f"{score_name_prefix}{val_or_test}/best_f1_epoch": best_f1_epoch,
                    "epoch": n_epochs  # Log at final epoch
                })

        # Load best F1 model if requested (for future use)
        if track_best_f1 and best_model_state and save_best_model_path and load_best_model:
            best_model_checkpoint = torch_load(str(save_best_model_path) + ".best_f1")
            self.model.load_state_dict(best_model_checkpoint['model_state_dict'])
            self.maml.load_state_dict(best_model_checkpoint['maml_state_dict'])
            logger.info(f"Loaded best F1 model from epoch {best_f1_epoch+1}")

        return train_results, val_result
    def evaluate(
        self,
        dataloader: DataLoader,
        score_name_prefix: str,
        epoch: int = None,
        log_metrics: bool = True,
    ):
        """Evaluate the model on the entire validation dataset"""
        self.maml.eval()
        
        # Track and adjust dataloader states like in ProtoNet
        if isinstance(dataloader, DataLoader):
            dataloader_states = {}
            if hasattr(dataloader.batch_sampler, "training"):
                dataloader_states["training"] = dataloader.batch_sampler.training
                dataloader.batch_sampler.training = False
            if hasattr(dataloader.batch_sampler, "use_all_remaining"):
                dataloader_states["use_all_remaining"] = dataloader.batch_sampler.use_all_remaining
                dataloader.batch_sampler.use_all_remaining = True
        
        # data_iter = iter(dataloader)
        # Use MAML's evaluate_step for the core logic
        # all_batches = [task for batch in batch_tasks(data_iter, 1) for task in batch]
        results = self.evaluate_step(dataloader)

        # Log metrics like in ProtoNet
        if log_metrics and results:
            eval_log = {
                f"{score_name_prefix}/loss": results["loss"],
                f"{score_name_prefix}/accuracy": results["accuracy"],
                f"{score_name_prefix}/f1": results["f1"],
                f"{score_name_prefix}/precision": results["precision"],
                f"{score_name_prefix}/recall": results["recall"],
                f"{score_name_prefix}/roc_auc": results["roc_auc"],
                f"{score_name_prefix}/average_precision": results.get("average_precision", 0),
            }
            
            if epoch is not None:
                eval_log["epoch"] = epoch
            if log_step is not None:
                eval_log["log_step"] = log_step
                
            wandb.log(eval_log)
            
            # Print evaluation metrics like in ProtoNet
            logger.info(f"Evaluation after epoch {epoch} for {score_name_prefix}: Loss = {results['loss']:.4f}")
            logger.info(
                f"Accuracy = {results['accuracy']:.4f}, "
                + f"F1 = {results['f1']:.4f}, "
                + f"Precision = {results['precision']:.4f}, "
                + f"Recall = {results['recall']:.4f}, "
                + f"ROC-AUC = {results['roc_auc']:.4f}"
            )

        # Restore dataloader states
        if isinstance(dataloader, DataLoader) and dataloader_states:
            if "training" in dataloader_states:
                dataloader.batch_sampler.training = dataloader_states["training"]
            if "use_all_remaining" in dataloader_states:
                dataloader.batch_sampler.use_all_remaining = dataloader_states["use_all_remaining"]
        
        self.maml.train()
        if hasattr(dataloader.dataset, "training"):
            dataloader.dataset.training = dataloader_original_training_state
        if hasattr(dataloader.batch_sampler, "use_all_remaining"):
            dataloader.batch_sampler.use_all_remaining = dataloader_original_use_all_remaining_state
        return results
