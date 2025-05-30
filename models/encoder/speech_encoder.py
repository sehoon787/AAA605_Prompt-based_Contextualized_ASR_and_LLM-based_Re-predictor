import torch
import torch.nn as nn

class ZipformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_expansion=4, dropout=0.1, layer_scale_init_value=1e-5):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_expansion,
            dropout=dropout,
            batch_first=True
        )
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(hidden_dim))

    def forward(self, x):
        out = self.transformer(x)
        out = self.layer_scale * out
        return x + out

class ZipDownsampling(nn.Module):
    def __init__(self, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.pool = nn.AvgPool1d(
            kernel_size=reduction_factor,
            stride=reduction_factor
        )

    def forward(self, x):
        B, T, H = x.shape
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x

class SpeechEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, num_stages=4, blocks_per_stage=2, reduction_factor=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            blocks = nn.ModuleList([
                ZipformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=4,
                    ff_expansion=4,
                    dropout=0.1,
                    layer_scale_init_value=1e-5
                ) for _ in range(blocks_per_stage)
            ])
            downsample = ZipDownsampling(reduction_factor)
            self.stages.append(nn.ModuleDict({
                "blocks": blocks,
                "downsample": downsample
            }))

    def forward(self, x):
        x = self.input_proj(x)
        for stage in self.stages:
            for block in stage["blocks"]:
                x = block(x)
            x = stage["downsample"](x)
        return x
