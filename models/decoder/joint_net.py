import torch.nn as nn

class JointNet(nn.Module):
    def __init__(self, encoder_dim, predictor_dim, joint_dim, vocab_size):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
        self.predictor_proj = nn.Linear(predictor_dim, joint_dim)
        self.joint_activation = nn.Tanh()
        self.output_layer = nn.Linear(joint_dim, vocab_size)

    def forward(self, encoder_out, predictor_out):
        enc = self.encoder_proj(encoder_out).unsqueeze(2)  # (B, T, 1, H)
        pred = self.predictor_proj(predictor_out).unsqueeze(1)  # (B, 1, U, H)
        joint = self.joint_activation(enc + pred)
        logits = self.output_layer(joint)  # (B, T, U, V)
        return logits
