from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, sigmoid


def compute_metrics(predictions: Tensor, targets: Tensor):
    """
    Compute classification metrics: Accuracy, F1-score, Precision, Recall, and ROC-AUC.

    Args:
        predictions (torch.Tensor): Model logits (before sigmoid activation for binary classification).
        targets (torch.Tensor): Ground truth labels (binary: 0 or 1).

    Returns:
        dict: A dictionary containing accuracy, F1, precision, recall, and ROC-AUC scores.
    """
    predictions = predictions.detach()
    targets = targets.detach()
    # Convert logits to binary predictions
    probs = sigmoid(predictions)  # Apply sigmoid to get probabilities
    binary_preds = (
        (probs > 0.5).float().view(targets.shape)
    )  # Convert to binary predictions

    # Convert tensors to numpy for sklearn
    y_true = targets.cpu().numpy()
    y_pred = binary_preds.cpu().numpy()
    y_probs = probs.cpu().numpy()  # For ROC-AUC

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Handle case where all targets are the same (ROC-AUC cannot be computed)
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }
