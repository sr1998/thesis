from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch import nn, no_grad
from torch.utils.data import DataLoader

import wandb
from src.data.helper_functions import metalearning_binary_target_changer
from src.models.helper_functions import batch_tasks
from src.models.protonet import ProtoNet
from src.scoring.metalearning_scoring_fn import compute_metrics


# code from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial16/Meta_Learning.html    

class ProtoMAMLTrainer(nn.Module):
    """Adapter class that provides MAML-like interface for ProtoMAML model"""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        train_k_shot: int,
        inner_lr: float,
        num_inner_steps: int,
        lr_output: float,
        eval_k_shot: int = None,
        starting_lr: float = 0.01,
        scheduler_step: int = 50,
        scheduler_gamma: float = 0.5,
        weight_decay: float = 0.0,
    ):
        super(ProtoMAMLTrainer, self).__init__()
        # Store parameters
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.lr_output = lr_output
        self.device = device
        self.starting_lr = starting_lr
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.train_k_shot = train_k_shot
        self.eval_k_shot = eval_k_shot or train_k_shot
        self.weight_decay = weight_decay

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.starting_lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma,
        )

        self.model = model.to(self.device)

        # Training state
        self.current_epoch = 0

    def run_model(self, local_model, output_weight, output_bias, imgs, labels):
        # Execute a model with given output layer weights and inputs
        feats = local_model(imgs)
        preds = F.linear(feats, output_weight, output_bias)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc

    def adapt_few_shot(self, support_imgs, support_targets):
        # Determine prototype initialization
        support_feats = self.model(support_imgs)
        prototypes, classes = ProtoNet.calculate_prototypes(
            support_feats, support_targets
        )
        support_labels = (
            (classes[None, :] == support_targets[:, None]).long().argmax(dim=-1)
        )
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.inner_lr)
        local_optim.zero_grad()
        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -(torch.norm(prototypes, dim=1) ** 2)
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        # Optimize inner loop model on support set
        for _ in range(self.num_inner_steps):
            # Determine loss on the support set
            loss, _, _ = self.run_model(
                local_model, output_weight, output_bias, support_imgs, support_labels
            )
            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            # Update output layer via SGD
            # (https://discuss.pytorch.org/t/the-difference-between-torch-tensor-data-and-torch-tensor/25995/4):
            with torch.no_grad():
                output_weight.copy_(output_weight - self.lr_output * output_weight.grad)
                output_bias.copy_(output_bias - self.lr_output * output_bias.grad)

            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias, classes

    def outer_loop(
        self, batch, mode="train", log_gradients=False, score_name_prefix=None
    ):
        self.model.zero_grad()
        losses_list = []
        preds_list = []
        labels_list = []


        # Determine gradients for batch of tasks
        for X, y in batch:
            y = metalearning_binary_target_changer(y)
            X = X.to(self.device)
            y = y.to(self.device)

            X_support = X[: self.train_k_shot * 2, :]
            y_support = y[: self.train_k_shot * 2]
            X_query = X[self.train_k_shot * 2 :, :]
            y_query = y[self.train_k_shot * 2 :]

            # Perform inner loop adaptation
            local_model, output_weight, output_bias, classes = self.adapt_few_shot(
                X_support, y_support
            )
            # Determine loss of query set
            query_labels = (classes[None, :] == y_query[:, None]).long().argmax(dim=-1)
            loss, preds, acc = self.run_model(
                local_model, output_weight, output_bias, X_query, query_labels
            )
            losses_list.append(loss.item())
            preds_list.append(preds.argmax(dim=1))
            labels_list.append(y_query)
            # Calculate gradients for query set loss
            if mode == "train":
                loss.backward()

                for p_global, p_local in zip(
                    self.model.parameters(), local_model.parameters()
                ):
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model

        # Perform update of base model
        if mode == "train":
            if log_gradients:
                self._log_gradients(self.current_epoch, score_name_prefix or "")
            opt = self.optimizer
            opt.step()
            opt.zero_grad()

        return torch.mean(torch.tensor(losses_list)), torch.cat(preds_list), torch.cat(labels_list)

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()

    def _log_gradients(self, epoch, score_name_prefix):
        """Log gradient statistics to wandb"""
        if (epoch + 1) % 10 == 0:
            with no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # Log gradient statistics
                        gradient_log = {
                            f"{score_name_prefix}.gradients/{name}_norm": param.grad.norm().item(),
                            f"{score_name_prefix}.gradients/{name}_mean": param.grad.mean().item(),
                            f"{score_name_prefix}.gradients/{name}_max": param.grad.max().item(),
                            f"{score_name_prefix}.gradients/{name}_min": param.grad.min().item(),
                            f"{score_name_prefix}.gradients/{name}_histogram": wandb.Histogram(
                                param.grad.detach().cpu().numpy().flatten()
                            ),
                            "epoch": epoch + 1,
                        }

                        # Only calculate std if there are at least 2 elements
                        if param.grad.numel() > 1:
                            gradient_log[
                                f"{score_name_prefix}.gradients/{name}_std"
                            ] = param.grad.std().item()

                        wandb.log(gradient_log)

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
        log_gradients: bool = False,
        score_name_prefix: str = None,
        save_best_model_path: str = None,
        track_best_f1: bool = True,
        load_best_model: bool = False,
        **kwargs,  # to be ignored
    ):
        self.model.train()
        score_name_prefix = score_name_prefix + "." if score_name_prefix else ""
        best_metric_value = (
            float("inf") if "loss" in early_stopping_metric else -float("inf")
        )

        patience_counter = 0

        # Best F1 tracking
        best_f1 = -float("inf")
        best_f1_epoch = 0
        best_model_state = None

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}")
            # log every 10 epochs, overwriting previous log
            if log_metrics and epoch % 10 == 0:
                train_results = self.evaluate(
                    train_dataloader, f"{score_name_prefix}train", epoch, log_metrics
                )

            # Validation phase
            if eval_dataloader and (early_stopping_patience or track_best_f1):
                val_result = self.evaluate(
                    eval_dataloader,
                    f"{score_name_prefix}{val_or_test}",
                    epoch,
                    log_metrics=True if epoch % 10 == 0 and log_metrics else False,
                )

                # Track best F1 score
                if track_best_f1 and "f1" in val_result and val_result["f1"] > best_f1:
                    best_f1 = val_result["f1"]
                    best_f1_epoch = epoch

                    # Save model state with best F1
                    if save_best_model_path:
                        best_model_state = {
                            "state_dict": self.model.state_dict(),
                            "epoch": epoch,
                            "f1_score": best_f1,
                        }
                        torch.save(
                            best_model_state, str(save_best_model_path) + ".best_f1"
                        )
                        logger.info(
                            f"Saved new best F1 model with F1 = {best_f1:.4f} at epoch {epoch+1}"
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
                            torch.save(self, save_best_model_path)
                            logger.info(
                                f"Saved new best model with {early_stopping_metric} = {current_metric:.4f}"
                            )
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break

            self.current_epoch = epoch

            batches = batch_tasks(train_dataloader, n_parallel_tasks)
            for i, batch in enumerate(batches):
                self.outer_loop(batch, mode="train", log_gradients=log_gradients, score_name_prefix=score_name_prefix)

            self.step_scheduler()

        train_results = self.evaluate(
            train_dataloader, f"{score_name_prefix}train", n_epochs, log_metrics
        )

        # Validation phase
        val_result = self.evaluate(
            eval_dataloader,
            f"{score_name_prefix}{val_or_test}",
            n_epochs,
            log_metrics=log_metrics,
        )

        # Include best F1 information in results WITHOUT overriding original f1
        if track_best_f1:
            val_result["best_f1"] = best_f1
            val_result["best_f1_epoch"] = best_f1_epoch

            # Log the best F1 separately
            if log_metrics:
                import wandb

                wandb.log(
                    {
                        f"{score_name_prefix}{val_or_test}/best_f1": best_f1,
                        f"{score_name_prefix}{val_or_test}/best_f1_epoch": best_f1_epoch,
                        "epoch": n_epochs,  # Log at final epoch
                    }
                )

        # Load best F1 model if requested (for future use)
        if (
            track_best_f1
            and best_model_state
            and save_best_model_path
            and load_best_model
        ):
            best_model_checkpoint = torch.load(str(save_best_model_path) + ".best_f1")
            self.model.load_state_dict(best_model_checkpoint["state_dict"])
            self.current_epoch = best_model_checkpoint["epoch"]
            best_f1 = best_model_checkpoint["f1_score"]
            logger.info(
                f"Loaded best F1 model with F1 = {best_f1:.4f} from epoch {best_f1_epoch+1}"
            )

        return train_results, val_result

    def evaluate(
        self,
        dataloader: DataLoader,
        score_name_prefix: str,
        epoch: int = None,
        log_metrics: bool = True,
    ):
        self.model.eval()
        if hasattr(dataloader.dataset, "training"):
            dataloader_original_training_state = dataloader.dataset.training
            dataloader.dataset.training = False
        if hasattr(dataloader.batch_sampler, "use_all_remaining"):
            dataloader_original_use_all_remaining_state = (
                dataloader.batch_sampler.use_all_remaining
            )
            dataloader.batch_sampler.use_all_remaining = True

        results = {}
        batches = batch_tasks(dataloader, 1)
        for batch in batches:
            loss, y_hat, target_inds = self.outer_loop(batch, mode="eval")
            results["loss"] = loss.item()
            metrics = compute_metrics(y_hat, target_inds)
            for key in metrics:
                results[key] = float(metrics[key])

        if log_metrics and results:
            eval_log = {
                f"{score_name_prefix}/loss": results["loss"],
                f"{score_name_prefix}/accuracy": results["accuracy"],
                f"{score_name_prefix}/f1": results["f1"],
                f"{score_name_prefix}/precision": results["precision"],
                f"{score_name_prefix}/recall": results["recall"],
                f"{score_name_prefix}/roc_auc": results["roc_auc"],
                "epoch": epoch,
            }
            wandb.log(eval_log)

            logger.info(
                f"Evaluation after epoch {epoch} for {score_name_prefix}: Loss = {results['loss']:.2f}"
            )
            logger.info(
                f"Accuracy = {results['accuracy']:.2f}, "
                + f"F1 = {results['f1']:.2f}, "
                + f"Precision = {results['precision']:.2f}, "
                + f"Recall = {results['recall']:.2f}, "
                + f"ROC-AUC = {results['roc_auc']:.2f}"
            )
        self.model.train()
        if hasattr(dataloader.dataset, "training"):
            dataloader.dataset.training = dataloader_original_training_state
        if hasattr(dataloader.batch_sampler, "use_all_remaining"):
            dataloader.batch_sampler.use_all_remaining = (
                dataloader_original_use_all_remaining_state
            )
        return results
