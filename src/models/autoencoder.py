import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_inputs: int, hidden_units: list[int], n_outputs: int):
        self.layers = nn.ModuleList()
        dimensions = [n_inputs] + hidden_units + [n_outputs]

        for i in range(len(dimensions) - 1):
            layer = nn.Linear(dimensions[i], dimensions[i + 1])
            self.layers.append(layer)

        print(f"Instantiated {len(self.layers)} layers. Dimension list: {dimensions}")

    def forward(self, x):
        output = x
        for layer in self.layers[:-1]:
            output = layer(output)
            output = nn.ReLU(output)

        output = self.layers[-1](output)  # No relu on last layer

        return output
