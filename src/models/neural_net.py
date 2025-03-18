from importlib import import_module

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import Normalizer
from torch import Tensor, cat, device, float32, nn, no_grad, sigmoid, tensor
from torch.cuda import is_available
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import wandb
from src.models.models import HighlyFlexibleModel
from src.scoring.metalearning_scoring_fn import compute_metrics


class NeuralNetWrapper(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible wrapper for PyTorch neural network"""

    def __init__(
        self,
        n_epochs=5,
        batch_size=32,
        lr=1e-3,
        num_layers=2,
        layer_sizes=None,
        dropout_rate=0.5,
        layer_norm=True,
        batch_norm=False,
        activation="relu",
        X_eval=None,
        y_eval=None,
        score_name_prefix=None,
        val_or_test="val",
        log_metrics_every_n_epoch: int = 0,  # log metrics every n epochs. 0 means no logging
        log_gradients_every_n_epoch: int = 0,  # log gradients every n epochs. 0 means no logging
    ):
        self.n_input = None
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.device = device("cuda" if is_available() else "cpu")
        self._is_fitted = False
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.score_name_prefix = score_name_prefix
        self.val_or_test = val_or_test
        self.log_metrics_every_n_epoch = log_metrics_every_n_epoch
        self.log_gradients_every_n_epoch = log_gradients_every_n_epoch

    def _log_gradients(self, epoch, score_name_prefix):
        # Log gradients if needed
        if (
            self.log_gradients_every_n_epoch > 0
            and (epoch + 1) % self.log_gradients_every_n_epoch == 0
        ):
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

    def fit(
        self,
        X,
        y,
    ):
        """Fit the model to training data with optional validation (if set during initialization).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns:
        --------
        self : object
            Returns self (for sklearn compatibility)
        """
        score_name_prefix = (
            self.score_name_prefix + "." if self.score_name_prefix else ""
        )

        self.classes_ = np.unique(y)
        self.n_input = X.shape[1]
        self.model = HighlyFlexibleModel(
            n_input=self.n_input,
            num_layers=self.num_layers,
            layer_sizes=self.layer_sizes,
            dropout_rate=self.dropout_rate,
            layer_norm=self.layer_norm,
            batch_norm=self.batch_norm,
            activation=self.activation,
        )
        self.model.to(self.device, dtype=float)
        self.model.train()

        dataset = TensorDataset(tensor(np.array(X)), tensor(np.array(y)))
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=self.batch_size, drop_last=True
        )
        if self.X_eval is not None and self.y_eval is not None:
            dataset = TensorDataset(
                tensor(np.array(self.X_eval)), tensor(np.array(self.y_eval))
            )
            val_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=SequentialSampler(dataset),
                shuffle=False,
            )

        # Training setup
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        # wandb.watch(
        #     self.model,
        #     log="gradients",
        #     log_freq=max(1, self.batch_size * self.log_gradients_every_n_epoch),
        #     log_graph=True,
        # )

        # Training loop
        for epoch in range(self.n_epochs):
            # Calculate and log metrics at the end of each logging epoch
            if self.log_metrics_every_n_epoch > 0 and (
                (epoch + 1) % self.log_metrics_every_n_epoch == 0 or epoch == 0
            ):
                # Log training metrics using current model state
                self.evaluate(
                    dataloader,
                    epoch=epoch + 1,
                    score_name_prefix=score_name_prefix + "train",
                )

                # Log validation metrics if validation data is available
                if self.X_eval is not None and self.y_eval is not None:
                    self.evaluate(
                        val_dataloader,
                        epoch=epoch + 1,
                        score_name_prefix=score_name_prefix + self.val_or_test,
                    )

            epoch_loss = 0.0
            all_predictions = []
            all_targets = []

            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device, dtype=float)
                y_batch = y_batch.to(self.device, dtype=float)

                optimizer.zero_grad()

                logits = self.model(X_batch).view(-1)
                loss = loss_fn(logits, y_batch.view(-1))

                # Track metrics
                epoch_loss += loss.item()
                all_predictions.append(logits.detach())
                all_targets.append(y_batch.view(-1).detach())

                loss.backward()

                self._log_gradients(epoch, score_name_prefix)

                optimizer.step()

        # Log training metrics using current model state
        self.evaluate(
            dataloader,
            epoch=epoch + 1,
            score_name_prefix=score_name_prefix + "train",
        )

        # Log validation metrics if validation data is available
        if self.X_eval is not None and self.y_eval is not None:
            self.evaluate(
                val_dataloader,
                epoch=epoch + 1,
                score_name_prefix=score_name_prefix + self.val_or_test,
            )
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        self.model.eval()
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")

        X = tensor(np.array(X))

        # Check if the model contains BatchNorm layers
        has_batchnorm = any(
            isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d)
            for m in self.model.modules()
        )

        # Process data in batches, handling batch sizes of 1 specially for BatchNorm if needed
        all_probs = []
        with no_grad():
            for i in range(0, len(X), self.batch_size):
                end = min(i + self.batch_size, len(X))
                X_batch = X[i:end]
                X_batch = X_batch.to(self.device, dtype=float)

                # If batch size is 1 and we have BatchNorm, duplicate to create batch of size 2
                if has_batchnorm and X_batch.size(0) == 1:
                    X_batch = X_batch.repeat(2, 1)
                    logits = self.model(X_batch).squeeze()
                    # Take only the first (original) sample's prediction
                    prob = sigmoid(logits[0])
                    all_probs.append(prob.item())
                else:
                    logits = self.model(X_batch).squeeze()
                    probs = sigmoid(logits)

                    # Handle different output shapes
                    if probs.ndim == 0:  # Single tensor scalar
                        all_probs.append(probs.item())
                    else:  # Multiple values
                        all_probs.extend(probs.cpu().numpy().tolist())

        # Format as scikit-learn compatible 2D array with columns [prob_class_0, prob_class_1]
        all_probs = np.array(all_probs)
        return np.column_stack((1 - all_probs, all_probs))

    def predict(self, X):
        self.model.eval()
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

    def evaluate(self, dataloader, score_name_prefix, log_metrics=True, epoch=None):
        """Evaluate the model on a dataset and optionally log metrics"""

        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0

        loss_fn = nn.BCEWithLogitsLoss()

        with no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device, dtype=float)
                y_batch = y_batch.to(self.device, dtype=float)
                # Forward pass
                logits = self.model(X_batch).view(-1)
                loss = loss_fn(logits, y_batch.view(-1))

                # Track predictions and targets
                all_predictions.append(logits)
                all_targets.append(y_batch.view(-1))
                total_loss += loss.item()

        if all_predictions:
            # Concatenate all batches
            all_predictions = cat(all_predictions)
            all_targets = cat(all_targets)

            # Calculate metrics
            metrics = compute_metrics(all_predictions, all_targets)
            metrics["loss"] = total_loss / len(dataloader)

            # Log metrics if requested
            if log_metrics and self.log_metrics_every_n_epoch > 0:
                # Log to wandb
                to_log = {
                    f"{score_name_prefix}/loss": metrics["loss"],
                    f"{score_name_prefix}/accuracy": metrics["accuracy"],
                    f"{score_name_prefix}/f1": metrics["f1"],
                    f"{score_name_prefix}/precision": metrics["precision"],
                    f"{score_name_prefix}/recall": metrics["recall"],
                    f"{score_name_prefix}/roc_auc": metrics["roc_auc"],
                }
                if epoch is not None:
                    to_log["epoch"] = epoch
                wandb.log(to_log)

            self.model.train()
            return metrics

        return None
