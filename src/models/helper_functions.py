from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


def batch_tasks(dataloader: DataLoader, batch_size: int):
    """Create batches of tasks from a dataloader."""
    batch = []
    for task in dataloader:
        batch.append(task)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Yield remaining tasks
        yield batch


def set_learning_rate(optimizer: Optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
