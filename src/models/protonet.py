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

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    if output.shape[1] == 2:
        output = output[:, 1:2]  # Take only positive class probability
    
    # Ensure target has same shape as output
    if target.dim() == 1:
        target = target.unsqueeze(1)
        
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


class DiceLoss(_Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss=False,
        from_logits=True,
        smooth: float = 0.0,
        ignore_index=None,
        eps=1e-7,
    ):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = torch.tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = None

        # if self.mode == BINARY_MODE:
        #     y_true = y_true.view(bs, 1, -1)
        #     y_pred = y_pred.view(bs, 1, -1)

        #     if self.ignore_index is not None:
        #         mask = y_true != self.ignore_index
        #         y_pred = y_pred * mask
        #         y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()
    
ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky


# class FocalLoss(torch.nn.Module):
#     def __init__(self, alpha=None, gamma=2):
#         super().__init__()
#         self.alpha = alpha  # Can be tensor of per-class weights
#         self.gamma = gamma
    
#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
        
#         # Apply class-specific alpha if provided
#         if self.alpha is not None:
#             alpha_t = self.alpha[targets]
#             focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
#         else:
#             focal_loss = (1-pt)**self.gamma * ce_loss
            
#         return focal_loss.mean()
    
    
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class sigmoidF1(nn.Module):
    """from https://github.com/gabriben/metrics-as-losses/blob/main/VLAP/pytorchLosses.py"""
    def __init__(self, S = -1, E = 0):
        super(sigmoidF1, self).__init__()
        self.S = S
        self.E = E

    @torch.cuda.amp.autocast()
    def forward(self, y_hat, y):
        
        y_hat = torch.sigmoid(y_hat)

        b = torch.tensor(self.S)
        c = torch.tensor(self.E)

        sig = 1 / (1 + torch.exp(b * (y_hat + c)))

        tp = torch.sum(sig * y, dim=0)
        fp = torch.sum(sig * (1 - y), dim=0)
        fn = torch.sum((1 - sig) * y, dim=0)

        sigmoid_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - sigmoid_f1
        macroCost = torch.mean(cost)

        return macroCost
    

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
        self.dice_loss = DiceLoss(mode="binary")
        self.focal_tversky_loss = FocalTverskyLoss()
        self.focal_loss = FocalLoss()
        self.sigmoid_f1 = sigmoidF1()

    @staticmethod
    def calculate_prototypes_robust(features, targets, alpha=0.9):
        """
        Calculate class prototypes with moving average for stability,
        properly detached from computation graph
        
        Args:
            features: Feature vectors [N, feature_dim]
            targets: Class labels [N]
            alpha: Moving average factor (higher means more weight to previous prototypes)
        """
        classes, _ = torch.unique(targets).sort()
        
        # Initialize prototypes dictionary if not already created
        if not hasattr(ProtoNet, 'prototype_dict'):
            ProtoNet.prototype_dict = {}
        
        prototypes = []
        for c in classes:
            # Calculate current prototype for this class
            c_idx = torch.where(targets == c)[0]
            if len(c_idx) == 0:
                continue
                
            # Compute current prototype and detach from computation graph
            current_prototype = features[c_idx].mean(dim=0)
            
            # Apply moving average if we have a previous prototype
            c_key = c.item()  # Convert tensor to Python scalar for dict key
            if c_key in ProtoNet.prototype_dict:
                # Get previous prototype (already detached)
                prev_prototype = ProtoNet.prototype_dict[c_key]
                
                # Compute smoothed prototype and detach from computation graph
                smooth_prototype = alpha * prev_prototype + (1 - alpha) * current_prototype.detach()
                
                # Store the detached value
                ProtoNet.prototype_dict[c_key] = smooth_prototype.detach()
            else:
                # First time seeing this class, store detached prototype
                ProtoNet.prototype_dict[c_key] = current_prototype.detach()
                smooth_prototype = current_prototype  # Keep in computation graph for current iteration
                
            # For the current forward pass, use the prototype WITH gradient connections
            # This ensures gradient flow during the current iteration
            if c_key in ProtoNet.prototype_dict:
                # Use stored prototype as starting point but allow gradients for this iteration
                stored_prototype = ProtoNet.prototype_dict[c_key]
                # Create a new tensor with the same values but that requires grad
                smooth_prototype = alpha * stored_prototype + (1 - alpha) * current_prototype
            else:
                smooth_prototype = current_prototype
                
            prototypes.append(smooth_prototype)
        
        if not prototypes:  # Handle edge case
            return None, classes
            
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes, classes

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
        return preds, labels, dist
    
    def classify_feats_with_cosine_similarity(self, prototypes, classes, feats, targets):
        # Use cosine similarity instead of Euclidean distance
        # Normalize the feature vectors and prototypes
        feats_norm = F.normalize(feats, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarity = torch.matmul(feats_norm, prototypes_norm.t())
        
        # Convert similarity to log probabilities
        preds = F.log_softmax(similarity, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
    
        return preds, labels, None

    def loss(self, X_support, X_query, y_support, y_query, class_weights_loss_fn):
        # Determine training loss for a given support and query set
        support_feats = self.encoder(X_support)
        query_feats = self.encoder(X_query)
        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, y_support)
        preds, labels, dist = self.classify_feats(prototypes, classes, query_feats, y_query)
        loss = F.cross_entropy(preds, labels, weight=class_weights_loss_fn)
        # loss = self.dice_loss(preds, labels)
        # loss = self.focal_tversky_loss(preds, labels)
        # loss = F.nll_loss(preds, labels)  # Negative log likelihood loss
        # loss = self.focal_loss(preds, labels)

        # Sigmoid F1 loss
        # probs = F.softmax(-dist, dim=1)  # Convert distances to probabilities
        # y_one_hot = F.one_hot(labels, num_classes=len(classes)).float()
        # loss = self.sigmoid_f1(probs, y_one_hot)
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
        class_weights_loss_fn = (.5, .5),
        evaluate_every: int = 10,
    ):

        # Store parameters
        self.device = device
        self.starting_lr = starting_lr
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.train_k_shot = train_k_shot
        self.eval_k_shot = eval_k_shot or train_k_shot
        self.weight_decay = weight_decay
        self.class_weights_loss_fn = torch.tensor(class_weights_loss_fn).to(device)

        # Store model and configuration
        self.protonet = ProtoNet(model).to(device)

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
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.scheduler_gamma, patience=self.scheduler_step)
        # self.last_lr = self.optimizer.param_groups[0]["lr"]

        self.evaluate_every = evaluate_every
        
    def _log_gradients(self, epoch, score_name_prefix):
        """Log gradient statistics to wandb"""
        if (epoch + 1) % self.evaluate_every == 0:
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
        val_dataloader: DataLoader = None,
        eval_dataloader: DataLoader = None,
        val_or_test: str = "val",
        early_stopping_patience: int = None,
        early_stopping_metric: str = "loss",
        early_stopping_fraction: float = 0.0,
        log_metrics: bool = True,
        log_gradients: bool = False,
        score_name_prefix: str = None,
        save_best_model_path: str = None,
        track_best_f1: bool = True,
        load_best_model: bool = False,
        accumulation_steps: int = 50,
        **kwargs,  # to sbe ignored
    ):
        self.protonet.train()
        score_name_prefix = score_name_prefix + "." if score_name_prefix else ""
        best_metric_value = (
            float("inf") if "loss" in early_stopping_metric else -float("inf")
        )

        patience_counter = 0
        n_es = max(1, int(len(train_dataloader) * early_stopping_fraction))


        # Best F1 tracking
        best_f1 = -float("inf")
        best_f1_epoch = 0
        best_model_state = None
        all_f1_scores = []

        for epoch in range(n_epochs):
            if epoch % self.evaluate_every == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}")
            # log every {self.evaluate_every} epochs, overwriting previous log
            if log_metrics and (epoch % self.evaluate_every == 0 or epoch <= 20):
                train_results = self.evaluate(
                    train_dataloader, f"{score_name_prefix}train", epoch, log_metrics
                )

            # Validation phase
            if eval_dataloader and (epoch % self.evaluate_every == 0 or epoch <= 20):
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
                    #         "state_dict": self.protonet.state_dict(),
                    #         "epoch": epoch,
                    #         "f1_score": best_f1,
                    #     }
                    #     torch_save(
                    #         best_model_state, str(save_best_model_path) + ".best_f1"
                    #     )
                    #     logger.info(
                    #         f"Saved new best F1 model with F1 = {best_f1:.4f} at epoch {epoch+1}"
                    #     )

            self.current_epoch = epoch
            train_iter = iter(train_dataloader)
            
            # Early stopping check
            if early_stopping_patience:# and val_dataloader:
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
                    # if save_best_model_path:
                    #     torch_save(self, save_best_model_path)
                    #     logger.info(
                    #         f"Saved new best model with {early_stopping_metric} = {current_metric:.4f}"
                    #     )
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            self.optimizer.zero_grad()
            # train_iter = iter(train_dataloader)
            for X, y in train_iter:
                X, y = X.to(self.device), y.to(self.device, dtype=torch.int64)
                X_support = X[: self.train_k_shot * 2, :]
                y_support = y[: self.train_k_shot * 2]
                X_query = X[self.train_k_shot * 2 :, :]
                y_query = y[self.train_k_shot * 2 :]

                loss, y_hat, target_inds = self.protonet.loss(X_support, X_query, y_support, y_query, self.class_weights_loss_fn)

                # # Normalize loss by accumulation steps to maintain the same scale
                # normalized_loss = loss / accumulation_steps
                # normalized_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.protonet.parameters(), max_norm=1.0)

                loss.backward()
                self.optimizer.step()

                # i += 1
                # if (i + 1) % accumulation_steps == 0 or i == len(train_dataloader) - 1:
                #     # Log gradients if enabled
                #     if log_gradients:
                #         self._log_gradients(epoch, f"{score_name_prefix}train")
                    
                #     # Perform optimization step with accumulated gradients
                #     self.optimizer.step()

                #     # Reset gradients for next accumulation
                #     self.optimizer.zero_grad()
                    
                # Log gradients if enabled
                if log_gradients:
                    self._log_gradients(epoch, f"{score_name_prefix}train")
                
                # Reset gradients for next accumulation
                self.optimizer.zero_grad()

            self.scheduler.step()
            # self.scheduler.step(current_metric)
            # if self.last_lr != self.optimizer.param_groups[0]["lr"]:
            #     self.last_lr = self.optimizer.param_groups[0]["lr"]
            #     logger.info(f"Learning rate changed to {self.last_lr:.10f}")

        train_results = self.evaluate(
            train_dataloader, f"{score_name_prefix}train", epoch+1, log_metrics
        )

        # Validation phase
        val_result = self.evaluate(
            eval_dataloader,
            f"{score_name_prefix}{val_or_test}",
            epoch+1,
            log_metrics=log_metrics,
        )

        all_f1_scores.append(val_result["f1"])

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
        # if (
        #     track_best_f1
        #     and best_model_state
        #     and save_best_model_path
        #     and load_best_model
        # ):
        #     best_model_checkpoint = torch_load(str(save_best_model_path) + ".best_f1")
        #     self.protonet.load_state_dict(best_model_checkpoint["state_dict"])
        #     self.current_epoch = best_model_checkpoint["epoch"]
        #     best_f1 = best_model_checkpoint["f1_score"]
        #     logger.info(
        #         f"Loaded best F1 model with F1 = {best_f1:.4f} from epoch {best_f1_epoch+1}"
        #     )

        val_result["averaged_f1_score"] = sum(all_f1_scores[-10:]) / 10
        return train_results, val_result

    def evaluate(
        self,
        dataloader: DataLoader,
        score_name_prefix: str,
        epoch: int = None,
        log_metrics: bool = True,
    ):
        self.protonet.eval()
        if isinstance(dataloader, DataLoader):
            if hasattr(dataloader.batch_sampler, "training"):
                dataloader_original_training_state = dataloader.batch_sampler.training
                dataloader.batch_sampler.training = False
            if hasattr(dataloader.batch_sampler, "use_all_remaining"):
                dataloader_original_use_all_remaining_state = dataloader.batch_sampler.use_all_remaining
                dataloader.batch_sampler.use_all_remaining = True

        results = {}
        # task_results = {}
        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):
                # task_results[str(i)] = {}
                X, y = X.to(self.device), y.to(self.device, dtype=torch.int64)
                # task_results[str(i)]["n_samples"] = X.shape[0]
                X_support = X[: self.eval_k_shot * 2, :] if not "train" in score_name_prefix and "_es" not in score_name_prefix else X[: self.train_k_shot * 2, :]
                y_support = y[: self.eval_k_shot * 2] if not "train" in score_name_prefix and "_es" not in score_name_prefix else y[: self.train_k_shot * 2]
                X_query = X[self.eval_k_shot * 2 :, :] if not "train" in score_name_prefix and "_es" not in score_name_prefix else X[self.train_k_shot * 2 :, :]
                y_query = y[self.eval_k_shot * 2 :] if not "train" in score_name_prefix and "_es" not in score_name_prefix else y[self.train_k_shot * 2 :]

                loss, y_hat, target_inds = self.protonet.loss(X_support, X_query, y_support, y_query, self.class_weights_loss_fn)
                results["loss"] = loss.item()
                # task_results[str(i)][f"{score_name_prefix}/loss"] = results["loss"]
                metrics = compute_metrics(y_hat, target_inds)
                for key in metrics:
                    results[key] = float(metrics[key])
                    # task_results[str(i)][f"{score_name_prefix}/{key}"] = results[key]
                # task_results[str(i)]["epoch"] = epoch

        if log_metrics and results:
            eval_log = {
                f"{score_name_prefix}/loss": results["loss"],
                f"{score_name_prefix}/accuracy": results["accuracy"],
                f"{score_name_prefix}/f1": results["f1"],
                f"{score_name_prefix}/precision": results["precision"],
                f"{score_name_prefix}/recall": results["recall"],
                f"{score_name_prefix}/roc_auc": results["roc_auc"],
                f"{score_name_prefix}/average_precision": results["average_precision"],
                "epoch": epoch,
            }
            wandb.log(eval_log)
            # wandb.log(task_results)

            logger.info(
                f"Evaluation after epoch {epoch} for {score_name_prefix}: Loss = {results['loss']:.2f}"
            )
            logger.info(
                f"Accuracy = {results['accuracy']:.2f}, "
                + f"F1 = {results['f1']:.2f}, "
                + f"Precision = {results['precision']:.2f}, "
                + f"Recall = {results['recall']:.2f}, "
                + f"ROC-AUC = {results['roc_auc']:.2f}"
                + f"average_precision = {results['average_precision']:.2f}"
            )
        self.protonet.train()
        if isinstance(dataloader, DataLoader):
            if hasattr(dataloader.batch_sampler, "training"):
                dataloader.batch_sampler.training = dataloader_original_training_state
            if hasattr(dataloader.batch_sampler, "use_all_remaining"):
                dataloader.batch_sampler.use_all_remaining = dataloader_original_use_all_remaining_state
        return results
