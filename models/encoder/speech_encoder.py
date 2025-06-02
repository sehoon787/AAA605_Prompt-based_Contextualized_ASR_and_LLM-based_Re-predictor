import torch.nn as nn

from models.encoder.zipformer_block import BiasNorm, Bypass, ZipformerBlock

class ConvEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # 3단계 2D Convolution
        self.conv2d_1 = nn.Conv2d(1, 8, kernel_size=(3,3), stride=(1,2), padding=(1,1))
        self.conv2d_2 = nn.Conv2d(8, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.conv2d_3 = nn.Conv2d(32, 128, kernel_size=(3,3), stride=(1,2), padding=(1,1))

        # ConvNeXt 스타일 Block
        self.dwconv = nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128)  # depthwise
        self.pwconv1 = nn.Conv2d(128, 384, kernel_size=1)
        self.act = nn.SiLU()  # SwooshL 대신 SiLU 사용 (SwooshL은 논문 고유 activation, 대체 가능)
        self.pwconv2 = nn.Conv2d(384, 128, kernel_size=1)
        self.layer_norm = nn.LayerNorm(128)

        # 마지막 projection
        self.linear = nn.Linear(128, hidden_dim)
        self.bias_norm = BiasNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        x = x.unsqueeze(1)  # (B, 1, T, F)

        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)

        residual = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x + residual

        # (B, C, T, F) → (B, T, F, C)
        x = x.permute(0, 2, 3, 1)

        # flatten freq axis
        B, T, F, C = x.shape
        x = x.reshape(B, T, F * C)

        # projection
        x = self.linear(x)
        x = self.bias_norm(x)

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
                 zipformer_blocks=None,
                 reduction_factors=None,
                 num_heads=4,
                 ff_expansion=4,
                 dropout=0.1):
        super().__init__()

        if zipformer_blocks is None:
            zipformer_blocks = [2, 2, 3, 4, 3, 2]
        if reduction_factors is None:
            reduction_factors = [2, 3, 4, 3, 2]

        self.conv_embed = ConvEmbedding(input_dim, hidden_dim)
        self.stages = nn.ModuleList()
        self.bypass_modules = nn.ModuleList()

        num_stages = len(zipformer_blocks)
        for stage_idx in range(num_stages):
            blocks = nn.ModuleList([
                ZipformerBlock(hidden_dim, num_heads, ff_expansion, dropout)
                for _ in range(zipformer_blocks[stage_idx])
            ])

            if stage_idx == 0:
                # 첫 stage: downsample 없음
                self.stages.append(nn.ModuleDict({
                    "blocks": blocks
                }))
            else:
                # 이후 stage: downsample / upsample / bypass 포함
                downsample = Downsample(hidden_dim, reduction_factors[stage_idx-1])
                upsample = Upsample(hidden_dim, reduction_factors[stage_idx-1])
                bypass = Bypass(hidden_dim)

                self.stages.append(nn.ModuleDict({
                    "blocks": blocks,
                    "downsample": downsample,
                    "upsample": upsample
                }))
                self.bypass_modules.append(bypass)

    def forward(self, x):
        x = self.conv_embed(x)

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 0:
                for block in stage["blocks"]:
                    x = block(x)
            else:
                residual = x

                x = stage["downsample"](x)
                for block in stage["blocks"]:
                    x = block(x)
                x = stage["upsample"](x)
                x = self.bypass_modules[stage_idx-1](residual, x)

        return x
