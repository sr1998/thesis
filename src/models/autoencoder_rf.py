import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.2):
        super(Encoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create encoder layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Final layer for latent representation
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(Decoder, self).__init__()
        
        hidden_dims = hidden_dims[::-1]  # Reverse hidden dimensions for decoder
        
        layers = []
        prev_dim = latent_dim
        
        # Create decoder layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Final layer to output reconstruction
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, dropout_rate)
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x):
        return self.encoder(x)

class AutoencoderRFProtoNet:
    """A simplified replacement for ProtoNet that uses an autoencoder for feature extraction and RF for classification"""
    
    def __init__(
        self,
        model=None,  # Not used but kept for API compatibility
        *,
        device,
        train_k_shot=None,  # Not used but kept for API compatibility
        eval_k_shot=None,   # Not used but kept for API compatibility
        latent_dim=128,
        hidden_dims=[512, 256],
        autoencoder_epochs=100,
        autoencoder_batch_size=64,
        starting_lr=0.001,
        weight_decay=0.0001,
        dropout_rate=0.2,
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        jitter_fraction=0.0,  # Optional data augmentation
        **kwargs  # Accept any additional parameters for compatibility
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.autoencoder_epochs = autoencoder_epochs
        self.autoencoder_batch_size = autoencoder_batch_size
        self.starting_lr = starting_lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.jitter_fraction = jitter_fraction
        
        # RF classifier parameters
        self.rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'class_weight': class_weight,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.autoencoder = None
        self.rf_classifier = None
        
    def fit(
        self,
        *,
        train_dataloader,
        n_epochs=None,  # Ignored, using autoencoder_epochs instead
        val_dataloader=None,
        eval_dataloader=None,
        val_or_test="val",
        log_metrics=True,
        **kwargs  # Catch other arguments for compatibility
    ):
        """Train the autoencoder and RF model on the given data"""
        logger.info("Training AutoencoderRF model")
        
        # Extract all data from train_dataloader
        X_train, y_train = self._extract_all_data_from_loader(train_dataloader)
        
        # Train the autoencoder on training data
        self._train_autoencoder(X_train)
        
        # Apply jitter if specified (data augmentation)
        if self.jitter_fraction > 0:
            X_train, y_train = self._apply_jitter(X_train, y_train)
        
        # Transform data using autoencoder
        X_train_encoded = self._encode_features(X_train)
        
        # Train RF on encoded features
        logger.info("Training Random Forest classifier")
        self.rf_classifier = RandomForestClassifier(**self.rf_params)
        self.rf_classifier.fit(X_train_encoded, y_train)
        
        # Evaluate and return results
        results = {}
        
        # Evaluate on training set
        train_metrics = self._evaluate_data(X_train, y_train, "train", log_metrics)
        results.update(train_metrics)
        
        # Evaluate on validation set if provided
        if val_dataloader:
            X_val, y_val = self._extract_all_data_from_loader(val_dataloader)
            val_metrics = self._evaluate_data(X_val, y_val, "val", log_metrics)
            results.update(val_metrics)
            
        # Evaluate on test set if provided
        if eval_dataloader:
            X_test, y_test = self._extract_all_data_from_loader(eval_dataloader)
            test_metrics = self._evaluate_data(X_test, y_test, val_or_test, log_metrics)
            results.update(test_metrics)
            
        return results
    
    def _extract_all_data_from_loader(self, dataloader):
        """Extract all data from a dataloader"""
        all_features = []
        all_labels = []
        
        for batch in dataloader:
            if isinstance(batch, tuple) and len(batch) >= 2:
                X, y = batch
                all_features.append(X)
                all_labels.append(y)
            else:
                # Handle case where batch doesn't have expected structure
                logger.warning("Unexpected batch structure in dataloader")
                
        # Convert to numpy arrays for sklearn compatibility
        if len(all_features) > 0:
            X = torch.cat(all_features).numpy() if torch.is_tensor(all_features[0]) else np.vstack(all_features)
            y = torch.cat(all_labels).numpy() if torch.is_tensor(all_labels[0]) else np.concatenate(all_labels)
            return X, y
        return None, None

    def _apply_jitter(self, X, y):
        """Apply random jitter to augment data"""
        X_jittered = X.copy()
        std_dev = np.std(X, axis=0) * self.jitter_fraction
        
        # Create jittered copies
        num_jittered = int(len(X) * 0.5)  # Add 50% more samples
        indices = np.random.choice(len(X), num_jittered, replace=True)
        X_new = X[indices] + np.random.normal(0, std_dev, (num_jittered, X.shape[1]))
        y_new = y[indices]
        
        # Combine original and jittered data
        X_combined = np.vstack([X, X_new])
        y_combined = np.concatenate([y, y_new])
        
        # Shuffle the combined data
        shuffle_idx = np.random.permutation(len(X_combined))
        return X_combined[shuffle_idx], y_combined[shuffle_idx]
        
    def _train_autoencoder(self, X_train):
        """Train an autoencoder on the input data"""
        input_dim = X_train.shape[1]
        
        # Create the autoencoder
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Create DataLoader for autoencoder training
        tensor_X = torch.FloatTensor(X_train).to(self.device)
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_X)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.autoencoder_batch_size, 
            shuffle=True
        )
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.autoencoder.parameters(),
            lr=self.starting_lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Train autoencoder
        self.autoencoder.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.autoencoder_epochs):
            total_loss = 0.0
            for x_batch, _ in dataloader:
                # Forward pass
                recon, _ = self.autoencoder(x_batch)
                loss = criterion(recon, x_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * x_batch.size(0)
                
            avg_epoch_loss = total_loss / len(tensor_X)
            scheduler.step(avg_epoch_loss)
            
            # Log every 10 epochs
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.autoencoder_epochs}, Loss: {avg_epoch_loss:.6f}")
            
            # Simple early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:  # Early stopping after 15 epochs without improvement
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
        logger.info("Autoencoder training completed")
        
    def _encode_features(self, X):
        """Encode features using the trained autoencoder"""
        self.autoencoder.eval()
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(self.device)
            encoded = self.autoencoder.encode(tensor_X).cpu().numpy()
        return encoded
    
    def _evaluate_data(self, X, y, score_name_prefix, log_metrics):
        """Evaluate the model on the given data"""
        # Encode the features
        X_encoded = self._encode_features(X)
        
        # Make predictions
        y_pred = self.rf_classifier.predict(X_encoded)
        y_pred_proba = self.rf_classifier.predict_proba(X_encoded)[:, 1]
        
        # Calculate metrics
        metrics = {
            f"{score_name_prefix}/accuracy": accuracy_score(y, y_pred),
            f"{score_name_prefix}/precision": precision_score(y, y_pred, zero_division=0),
            f"{score_name_prefix}/recall": recall_score(y, y_pred, zero_division=0),
            f"{score_name_prefix}/f1": f1_score(y, y_pred, zero_division=0),
            f"{score_name_prefix}/best_f1": f1_score(y, y_pred, zero_division=0),
            f"{score_name_prefix}/roc_auc": roc_auc_score(y, y_pred_proba),
            f"{score_name_prefix}/loss": 1 - f1_score(y, y_pred, zero_division=0),
        }
        
        if log_metrics:
            try:
                import wandb
                wandb.log(metrics)
            except ImportError:
                logger.info(f"Evaluation results: {metrics}")
        
        return metrics

    def evaluate(self, dataloader, score_name_prefix, epoch=None, log_metrics=True):
        """Compatible API with ProtoNet for evaluation"""
        X, y = self._extract_all_data_from_loader(dataloader)
        return self._evaluate_data(X, y, score_name_prefix, log_metrics)