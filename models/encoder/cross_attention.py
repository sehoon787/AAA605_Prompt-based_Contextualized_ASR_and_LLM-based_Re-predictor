import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, audio_dim, text_dim, fusion_dim):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            batch_first=True
        )
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

    def forward(self, audio, text):
        audio_proj = self.audio_proj(audio)
        text_proj = self.text_proj(text)
        output, _ = self.cross_attention(audio_proj, text_proj, text_proj)
        return output
