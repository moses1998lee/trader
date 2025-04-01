# %%
import torch
import torch.nn as nn


class SimpleLNN(nn.Module):
    def __init__(
        self,
        x_t_size: int,
        hidden_size: list[int],
        dx_dt_size: int,
        time_constant: float,
    ):
        super().__init__()
        self.input_size = x_t_size
        self.hidden_size = hidden_size
        self.output_size = dx_dt_size
        self.tau = time_constant
        
        self.fx = nn.Linear(x_t_size, hidden_size[0])
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(hidden_size[i], hidden_size[i + 1])
                for i in range(len(hidden_size) - 1)
            ]
        )

        
        self.A = nn.Parameter(torch.randn(dx_dt_size, dx_dt_size))
        self.output_layer = nn.Linear(hidden_size[-1], output_size)
    
    def forward(x_t):
        

    def set_time_constant(self, value: float):
        tmp = self.tau
        self.tau = value
        print(f"Tau was modified from {tmp} to {self.tau}!")
