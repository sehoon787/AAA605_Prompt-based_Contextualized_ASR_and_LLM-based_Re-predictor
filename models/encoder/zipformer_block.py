import torch
import torch.nn as nn
import torch.nn.functional as F

# BiasNorm: RMS-based normalization with bias
class BiasNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))  # log scale for stability
        self.eps = eps

    def forward(self, x):
        # x: (B, T, D)
        rms = torch.sqrt(torch.mean((x - self.bias) ** 2, dim=-1, keepdim=True) + self.eps)
        normalized = x / rms
        scaled = normalized * torch.exp(self.log_scale)
        return scaled

# Non-linear Attention block
class NonLinearAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, H)
        q = self.linear_q(x)  # (B, T, H)
        k = self.linear_k(x)  # (B, T, H)
        v = self.linear_v(x)  # (B, T, H)

        attn_scores = torch.tanh(q) * k  # (B, T, H)
        attn_weights = F.softmax(attn_scores, dim=1)  # attention over T (time)
        context = attn_weights * v  # (B, T, H)
        output = self.linear_out(context)  # (B, T, H)

        return output

# Convolution block with Depthwise + GLU
class ConvModule(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.norm = BiasNorm(hidden_dim)
        self.pointwise_conv1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)  # (B, T, H)
        x = x.transpose(1, 2)  # (B, H, T)

        x = self.pointwise_conv1(x)  # (B, 2H, T)
        x = self.glu(x)  # (B, H, T)

        x = self.depthwise_conv(x)  # (B, H, T)
        x = self.batch_norm(x)  # (B, H, T)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # (B, H, T)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # (B, T, H)
        return x

# y: (1 − c) ⊙ x + c ⊙ y
class Bypass(nn.Module):
    def __init__(self, dim, init_min=0.9, init_max=1.0):
        super().__init__()
        init_c = torch.empty(dim).uniform_(init_min, init_max)
        self.c = nn.Parameter(init_c)

    def forward(self, x, y):
        c = torch.clamp(self.c, 0.0, 1.0)  # 항상 [0,1] 범위로 제한
        return (1 - c) * x + c * y

# Final ZipformerBlock
class ZipformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_expansion=4, dropout=0.1, layer_scale_init_value=1e-5):
        super().__init__()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(hidden_dim))

        # Feedforward 1
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * ff_expansion, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn1 = BiasNorm(hidden_dim)

        # Non-linear Attention
        self.non_linear_attention = NonLinearAttention(hidden_dim)

        # Self-attention 1
        self.self_attn1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_attn1 = BiasNorm(hidden_dim)

        # Convolution 1
        self.conv1 = ConvModule(hidden_dim, kernel_size=3, dropout=dropout)

        # Feedforward 2
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * ff_expansion, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn2 = BiasNorm(hidden_dim)

        # Middle Bypass
        self.bypass_mid = Bypass(hidden_dim)

        # Self-attention 2
        self.self_attn2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_attn2 = BiasNorm(hidden_dim)

        # Convolution 2
        self.conv2 = ConvModule(hidden_dim, kernel_size=3, dropout=dropout)

        # Feedforward 3
        self.ffn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * ff_expansion, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn3 = BiasNorm(hidden_dim)

        # Final BiasNorm
        self.bias_norm_final = BiasNorm(hidden_dim)

        # Final Bypass
        self.bypass_final = Bypass(hidden_dim)

    def forward(self, x):
        residual = x

        # Feedforward 1
        x = x + self.layer_scale * self.norm_ffn1(self.ffn1(x))
        # Non-linear Attention
        x = x + self.layer_scale * self.non_linear_attention(x)
        # Self-attention 1
        attn_out1, _ = self.self_attn1(x, x, x)
        x = x + self.layer_scale * self.norm_attn1(attn_out1)
        # Convolution 1
        x = x + self.layer_scale * self.conv1(x)
        # Feedforward 2
        x = x + self.layer_scale * self.norm_ffn2(self.ffn2(x))
        # 중간 Bypass 적용
        c_mid = self.bypass_mid(residual, x)
        x = (1 - c_mid) * residual + c_mid * x
        # Self-attention 2
        attn_out2, _ = self.self_attn2(x, x, x)
        x = x + self.layer_scale * self.norm_attn2(attn_out2)
        # Convolution 2
        x = x + self.layer_scale * self.conv2(x)
        # Feedforward 3
        x = x + self.layer_scale * self.norm_ffn3(self.ffn3(x))
        # 최종 BiasNorm + 첫 입력 residual 더하기
        x = self.bias_norm_final(x)
        # 최종 Bypass 적용
        c_final = self.bypass_final(residual, x)
        x = (1 - c_final) * residual + c_final * x

        return x
