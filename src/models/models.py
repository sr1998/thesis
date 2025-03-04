from torch import Tensor, nn


class HighlyFlexibleModel(nn.Module):
    def __init__(
        self,
        n_input: int,
        num_layers: int = 2,
        layer_sizes: list = None,
        dropout_rate: float = 0.5,
        layer_norm: bool = True,
        batch_norm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        # Default layer sizes if none provided
        if layer_sizes is None:
            layer_sizes = [512, 256]
        elif len(layer_sizes) < num_layers:
            # Pad with default values if not enough sizes provided
            layer_sizes = layer_sizes + [256] * (num_layers - len(layer_sizes))

        # Select activation function
        if activation == "relu":
            self.act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU()
        elif activation == "elu":
            self.act_fn = nn.ELU()
        elif activation == "gelu":
            self.act_fn = nn.GELU()
        elif activation == "selu":
            self.act_fn = nn.SELU()
        else:
            self.act_fn = nn.ReLU()  # Default

        # Build network with dynamic number of layers
        layers = []
        in_features = n_input

        for i in range(num_layers):
            out_features = layer_sizes[i]

            # Add linear layer
            layers.append(nn.Linear(in_features, out_features))

            # Add normalization (either layer norm or batch norm, not both)
            if layer_norm:
                layers.append(nn.LayerNorm(out_features))
            elif batch_norm:
                layers.append(nn.BatchNorm1d(out_features))

            # Add activation and dropout
            layers.append(self.act_fn)
            layers.append(nn.Dropout(dropout_rate))

            in_features = out_features

        # Output layer (binary classification)
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Model1(nn.Module):
    def __init__(
        self,
        n_input: int,
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
            nn.Linear(n_input, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),  # Output layer for binary classification
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Model2(nn.Module):
    def __init__(self, n_input: int):
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


class Model3(nn.Module):
    def __init__(self, n_input: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def get_model(model_name: str) -> nn.Module:
    if model_name == "model1":
        return Model1
    elif model_name == "model2":
        return Model2
    elif model_name == "model3":
        return Model3
    else:
        raise ValueError(f"Model {model_name} not found.")
