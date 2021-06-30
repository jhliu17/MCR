import torch.nn as nn


class GatedTanh(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.in_layer = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Tanh())
        self.gate_layer = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Sigmoid())
    
    def forward(self, x):
        y = self.in_layer(x)
        g = self.gate_layer(x)
        y = y * g
        return y
