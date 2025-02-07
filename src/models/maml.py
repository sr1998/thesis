from copy import deepcopy

from loguru import logger
from torch import Tensor, nn, no_grad, zeros_like
from torch import device as torch_device
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(
        self,
        nodes_per_layer: list[int],
        activations_per_layer: list[nn.Module],
    ):
        super().__init__()
        # All layers should have an activation
        assert len(nodes_per_layer) - 1 == len(activations_per_layer)

        # Create neural net
        self.layers = []
        for i in range(len(nodes_per_layer) - 1):
            self.layers.append(nn.Linear(nodes_per_layer[i], nodes_per_layer[i + 1]))
            self.layers.append(activations_per_layer[i])

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Reptile:
    def __init__(
        self,
        model: nn.Module,
        n_gradient_steps: int,
        device: torch_device,
        loss_function: nn.Module,
        meta_optimizer: Optimizer,
        inner_lr: float,
    ):
        assert (
            next(model.parameters()).device == device
        ), "Model parameters are not on the specified device"

        self.model = model
        self.n_gradient_steps = n_gradient_steps
        self.loss_fn = loss_function  # TODO hypere_params?
        self.device = device
        self.outer_optimizer = meta_optimizer
        self.inner_lr = inner_lr

    def evaluate(self, dataloader: DataLoader, epoch: int):
        """
        Evaluate the current model on the full dataset.
        """
        global_model = self.model
        global_model.eval()
        correct = 0
        total = 0
        with no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = global_model(X_batch)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total * 100
        logger.info(f"Evaluation after epoch {epoch}: Accuracy = {accuracy:.2f}%")
        global_model.train()

    def fit(self, dataloader: DataLoader, n_epochs, n_parallel_tasks):
        global_model = self.model  # nice reference. Parameters are called phi here
        global_model.train()

        # we want each epoch to go through the whole dataset
        n_iters = n_epochs * (len(dataloader) // n_parallel_tasks + 1)
        epoch_count = 0

        task_iterator = iter(dataloader)

        for i in range(n_iters):
            # Initialize an accumulator for meta gradients (same shape as parameters)
            meta_gradients = {
                name: zeros_like(param.data)
                for name, param in global_model.named_parameters()
            }

            # n_parallel_batch models take self.n_gradient_steps, each for different tasks
            # updating a copy of global_model to parameters phi'. This gives a difference
            # phi - phi', contributing to model_param_deltas.
            for j in range(n_parallel_tasks):
                # get a task
                try:
                    X, y = next(task_iterator)
                except StopIteration:
                    # DataLoader is exhausted, meaning one epoch is done.
                    epoch_count += 1
                    logger.info(f"Epoch {epoch_count} complete at iteration {i+1}/{n_iters} with {n_parallel_tasks} parallel tasks and {len(dataloader)} total tasks. Reinitializing DataLoader for next epoch.")
                    self.evaluate(dataloader, epoch_count)
                    # prevents the model from training for more on some tasks than others
                    if epoch_count == n_epochs:
                        return
                    task_iterator = iter(dataloader)
                    X, y = next(task_iterator)
                X, y = X.to(self.device), y.to(self.device)

                # copy model and setup inner_optimizer
                copied_model = deepcopy(global_model)
                inner_optimizer = Adam(
                    copied_model.parameters(), lr=self.inner_lr
                )  # TODO hyper-params

                for k in range(self.n_gradient_steps):
                    inner_optimizer.zero_grad()
                    loss = self.loss_fn(copied_model(X), y)
                    loss.backward()
                    inner_optimizer.step()

                # Compute the pseudo-gradient: difference between the global model and the adapted model.
                # According to the paper, one may define the meta-gradient as (global - adapted)/inner_lr.
                with no_grad():
                    for (name, global_param), (_, copied_param) in zip(
                        global_model.named_parameters(), copied_model.named_parameters()
                    ):
                        # Accumulate the difference over tasks.
                        meta_gradients[name] += (
                            global_param.detach() - copied_param.detach()
                        )

            logger.info(f"Meta-update iteration {i+1}/{n_iters} complete: processed {n_parallel_tasks} parallel tasks.")

            # Average the accumulated meta gradients and scale by inner_lr.
            for name, param in global_model.named_parameters():
                meta_grad = meta_gradients[name] / n_parallel_tasks
                meta_grad = meta_grad / self.inner_lr
                param.grad = (
                    meta_grad  # Assign the pseudo-gradient for the meta optimizer
                )

            # Perform the meta-update using the outer optimizer.
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
