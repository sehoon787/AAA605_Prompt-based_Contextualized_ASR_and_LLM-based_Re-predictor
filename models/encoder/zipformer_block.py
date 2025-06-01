import torch
import torch.nn as nn
import torch.nn.functional as F


# BiasNorm 구현
class BiasNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.bias


# Non-linear Attention 구현
class NonLinearAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        attn_scores = torch.tanh(q + k)  # (B, T, H)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = attn_weights * v
        out = self.linear_out(context)
        return out


# Conv1D block
class ConvModule(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, T, H) -> (B, H, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x


# 전체 ZipformerBlock
class ZipformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_expansion=4, dropout=0.1, layer_scale_init_value=1e-5):
        super().__init__()

        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(hidden_dim))

        # 첫 Feed-Forward (Pre-attention)
        self.ff1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * ff_expansion, hidden_dim),
            nn.Dropout(dropout)
        )

        # Non-linear attention
        self.non_linear_attn = NonLinearAttention(hidden_dim)

        # Multi-head self attention
        self.self_attn1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Convolution module
        self.conv1 = ConvModule(hidden_dim, kernel_size=3, dropout=dropout)

        # Second Feed-Forward
        self.ff2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * ff_expansion, hidden_dim),
            nn.Dropout(dropout)
        )

        # Second self-attention
        self.self_attn2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Second convolution module
        self.conv2 = ConvModule(hidden_dim, kernel_size=3, dropout=dropout)

        # Third Feed-Forward
        self.ff3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * ff_expansion, hidden_dim),
            nn.Dropout(dropout)
        )

        # BiasNorm + Bypass 마지막
        self.bias_norm = BiasNorm(hidden_dim)

    def forward(self, x):
        residual = x

        x = x + self.layer_scale * self.ff1(x)
        x = x + self.layer_scale * self.non_linear_attn(x)
        x_attn1, _ = self.self_attn1(x, x, x)
        x = x + self.layer_scale * x_attn1
        x = x + self.layer_scale * self.conv1(x)
        x = x + self.layer_scale * self.ff2(x)
        x_attn2, _ = self.self_attn2(x, x, x)
        x = x + self.layer_scale * x_attn2
        x = x + self.layer_scale * self.conv2(x)
        x = x + self.layer_scale * self.ff3(x)

        # 마지막 BiasNorm + Bypass
        x = self.bias_norm(x)
        x = x + residual

        return x
