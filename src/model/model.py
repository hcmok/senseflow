import torch.nn as nn


class WordVectorFieldModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.hidden(x)
        out = self.output(h)
        return out
