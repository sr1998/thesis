from importlib import import_module

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import Normalizer
from torch import device, float32, nn, no_grad, sigmoid, tensor
from torch.cuda import is_available
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import wandb
from src.models.models import HighlyFlexibleModel


class NeuralNetWrapper(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible wrapper for PyTorch neural network"""

    def __init__(
        self,
        n_epochs=5,
        batch_size=32,
        lr=1e-3,
        scale_factor=100.0,
        num_layers=2,
        layer_sizes=None,
        dropout_rate=0.5,
        layer_norm=True,
        batch_norm=False,
        activation="relu",
        do_normalization=False,
    ):
        self.n_input = None
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scale_factor = scale_factor
        self.do_normalization = do_normalization
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.device = device("cuda" if is_available() else "cpu")
        self._is_fitted = False

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.n_input is None:
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
        X_batch = tensor(np.array(X))
        y_batch = tensor(np.array(y))
        X_batch = X_batch.to(self.device, dtype=float)
        y_batch = y_batch.to(self.device, dtype=float)

        dataset = TensorDataset(X_batch, y_batch)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        # Training setup
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        # Training loop
        for epoch in range(self.n_epochs):
            self.model.train()
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = loss_fn(logits.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")

        X = tensor(np.array(X))
        X = X.to(self.device, dtype=float)
        dataset = TensorDataset(X)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        # Prediction
        self.model.eval()
        all_probs = []

        with no_grad():
            for (X_batch,) in dataloader:
                logits = self.model(X_batch).squeeze()
                probs = sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().tolist())

        # Format as scikit-learn compatible 2D array with columns [prob_class_0, prob_class_1]
        all_probs = np.array(all_probs)
        return np.column_stack((1 - all_probs, all_probs))

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
