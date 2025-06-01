import torch.nn as nn

from models.encoder.zipformer_block import ZipformerBlock


class ConvEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, T, input_dim)
        x = x.transpose(1, 2)  # (B, input_dim, T)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = x.transpose(1, 2)  # (B, T, hidden_dim)
        return x

class Downsample(nn.Module):
    def __init__(self, hidden_dim, reduction_factor=2):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=reduction_factor, stride=reduction_factor)

    def forward(self, x):
        # x: (B, T, H)
        x = x.transpose(1, 2)  # (B, H, T)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (B, T_new, H)
        return x

class Upsample(nn.Module):
    def __init__(self, hidden_dim, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # 단순 repeat upsampling 후 projection (가장 간단한 방식)
        x = x.repeat_interleave(self.scale_factor, dim=1)
        x = self.linear(x)
        return x

class SpeechEncoder(nn.Module):
    def __init__(self,
                 input_dim=80,
                 hidden_dim=256,
                 num_stages=4,
                 blocks_per_stage=2,
                 reduction_factor=2,
                 num_heads=4,
                 ff_expansion=4,
                 dropout=0.1):
        super().__init__()

        self.conv_embed = ConvEmbedding(input_dim, hidden_dim)

        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            blocks = nn.ModuleList([
                ZipformerBlock(hidden_dim, num_heads, ff_expansion, dropout)
                for _ in range(blocks_per_stage)
            ])
            downsample = Downsample(hidden_dim, reduction_factor)
            upsample = Upsample(hidden_dim, reduction_factor)
            self.stages.append(nn.ModuleDict({
                "blocks": blocks,
                "downsample": downsample,
                "upsample": upsample
            }))

    def forward(self, x):
        x = self.conv_embed(x)
        residuals = []

        # Encoder stages
        for stage in self.stages:
            for block in stage["blocks"]:
                x = block(x)

            residuals.append(x)
            x = stage["downsample"](x)

        # 역으로 업샘플 + bypass 합산 (그림의 Zipformer 특징)
        for stage, residual in reversed(list(zip(self.stages, residuals))):
            x = stage["upsample"](x)
            x = x + residual  # bypass connection

        return x
