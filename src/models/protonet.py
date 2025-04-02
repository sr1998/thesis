import torch
import torch.nn.functional as F
from loguru import logger
from torch import device as torch_device
from torch import load as torch_load
from torch import nn, no_grad
from torch import save as torch_save
from torch.autograd import Variable
from torch.utils.data import DataLoader

import wandb
from src.models.helper_functions import batch_tasks, set_learning_rate
from src.scoring.metalearning_scoring_fn import compute_metrics


def euclidean_dist(x, y):
    # # Code taken from https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class ProtoNetOriginal(nn.Module):
    # # Code taken from https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    def __init__(self, encoder):
        super(ProtoNetOriginal, self).__init__()

        self.encoder = encoder

    def loss(self, X_support, X_query, y_support, y_query):
        xs = Variable(X_support)  # support
        xq = Variable(X_query)  # query

        n_class = 2  # Number of classes (2 for binary classification)
        n_total_support = X_support.size(0)  # Number of total support samples
        assert n_total_support % 2 == 0, "n_total_support must be even"
        n_support_per_class = n_total_support // 2  # Number of support samples per class

        target_inds = Variable(y_query, requires_grad=False).unsqueeze(1).long()

        if xq.is_cuda:
            print("CUDA")
            target_inds = target_inds.cuda()

        x = torch.cat([xs, xq], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[: n_class * n_support_per_class].view(n_class, n_support_per_class, z_dim).mean(1)
        zq = z[n_class * n_support_per_class :]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()
        
        _, y_hat = log_p_y.max(1)

        # acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        return loss_val, y_hat, target_inds.squeeze()


class ProtoNet(nn.Module):

    def __init__(self, encoder):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of Adam optimizer
        """
        super().__init__()
        self.encoder = encoder

    @staticmethod
    def calculate_prototypes(features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        classes, _ = torch.unique(targets).sort()  # Determine which classes we have
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes

    def classify_feats(self, prototypes, classes, feats, targets):
        # Classify new examples with prototypes and return classification error
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)  # Squared euclidean distance
        preds = F.log_softmax(-dist, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        # acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels

    def loss(self, X_support, X_query, y_support, y_query):
        # Determine training loss for a given support and query set
        support_feats = self.encoder(X_support)
        query_feats = self.encoder(X_query)
        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, y_support)
        preds, labels = self.classify_feats(prototypes, classes, query_feats, y_query)
        loss = F.cross_entropy(preds, labels)
        # loss = F.nll_loss(preds, labels)  # Negative log likelihood loss
        return loss, preds.argmax(dim=1), labels


class ProtonetTrainer:
    """Adapter class that provides MAML-like interface for Protonet model"""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch_device,
        train_k_shot: int,
        eval_k_shot: int = None,
        starting_lr: float = 0.01,
        scheduler_step: int = 50,
        scheduler_gamma: float = 0.5,
        weight_decay: float = 0.0,
    ):
        # Store model and configuration
        self.protonet = ProtoNet(model).to(device)

        # Store parameters
        self.device = device
        self.starting_lr = starting_lr
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.train_k_shot = train_k_shot
        self.eval_k_shot = eval_k_shot or train_k_shot
        self.weight_decay = weight_decay

        # Training state
        self.optimizer = None
        self.current_epoch = 0

        self.optimizer = torch.optim.Adam(
            self.protonet.parameters(),
            lr=self.starting_lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma,
        )

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()

    def _log_gradients(self, epoch, score_name_prefix):
        """Log gradient statistics to wandb"""
        if (epoch + 1) % 10 == 0:
            with no_grad():
                for name, param in self.protonet.named_parameters():
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
                            gradient_log[f"{score_name_prefix}.gradients/{name}_std"] = (
                                param.grad.std().item()
                            )

                        wandb.log(gradient_log)

    def fit(
        self,
        *,
        train_dataloader: DataLoader,
        n_epochs: int,
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
        self.protonet.train()
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
                            "state_dict": self.protonet.state_dict(),
                            "epoch": epoch,
                            "f1_score": best_f1,
                        }
                        torch_save(
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

            for i, (X, y) in enumerate(train_dataloader):
                X, y = X.to(self.device), y.to(self.device, dtype=torch.int64)
                X_support = X[: self.train_k_shot * 2, :]
                y_support = y[: self.train_k_shot * 2]
                X_query = X[self.train_k_shot * 2 :, :]
                y_query = y[self.train_k_shot * 2 :]

                loss, y_hat, target_inds = self.protonet.loss(X_support, X_query, y_support, y_query)
                loss.backward()
                self.optimizer.step()

                if log_gradients:
                    self._log_gradients(epoch, f"{score_name_prefix}train")

                self.optimizer.zero_grad()
            
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
            best_model_checkpoint = torch_load(str(save_best_model_path) + ".best_f1")
            self.protonet.load_state_dict(best_model_checkpoint["state_dict"])
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
        self.protonet.eval()
        if hasattr(dataloader.dataset, "training"):
            dataloader_original_training_state = dataloader.dataset.training
            dataloader.dataset.training = False
        if hasattr(dataloader.batch_sampler, "use_all_remaining"):
            dataloader_original_use_all_remaining_state = dataloader.batch_sampler.use_all_remaining
            dataloader.batch_sampler.use_all_remaining = True

        results = {}
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device, dtype=torch.int64)
            X_support = X[: self.train_k_shot * 2, :]
            y_support = y[: self.train_k_shot * 2]
            X_query = X[self.train_k_shot * 2 :, :]
            y_query = y[self.train_k_shot * 2 :]

            loss, y_hat, target_inds = self.protonet.loss(X_support, X_query, y_support, y_query)
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
        self.protonet.train()
        if hasattr(dataloader.dataset, "training"):
            dataloader.dataset.training = dataloader_original_training_state
        if hasattr(dataloader.batch_sampler, "use_all_remaining"):
            dataloader.batch_sampler.use_all_remaining = dataloader_original_use_all_remaining_state
        return results
