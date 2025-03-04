from random import randint

from torch import Tensor


def metalearning_binary_target_changer(labels: Tensor) -> Tensor:
    """Change the binary labels randomly.

    Args:
        labels: The binary labels to change.

    Returns:
        The changed binary labels.
    """
    to_change = randint(0, 1)
    labels = (labels + to_change) % 2
    return labels
