import torch.nn as nn

class TextAdapter(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, adapter_dim)
        )

    def forward(self, x):
        return self.adapter(x)
