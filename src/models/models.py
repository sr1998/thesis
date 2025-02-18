from torch import Tensor, nn

class Model1(nn.Module):
    def __init__(
        self,
        n_input: int
        # nodes_per_layer: list[int],
        # activations_per_layer: list[nn.Module],
    ):
        super().__init__()
        # All layers should have an activation
        # assert len(nodes_per_layer) - 1 == len(activations_per_layer)

        # Create neural net
        # self.layers = []
        # self.layers.append()
        # for i in range(len(nodes_per_layer) - 1):
        #     self.layers.append(nn.Linear(nodes_per_layer[i], nodes_per_layer[i + 1]))
        #     self.layers.append(activations_per_layer[i])

        self.layers = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),  # Output layer for binary classification
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
    

class Model2(nn.Module):
    def __init__(
        self,
        n_input: int
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),  # Output layer for binary classification
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
    
    
def get_model(model_name: str) -> nn.Module:
    if model_name == "model1":
        return Model1
    elif model_name == "model2":
        return Model2
    else:
        raise ValueError(f"Model {model_name} not found.")