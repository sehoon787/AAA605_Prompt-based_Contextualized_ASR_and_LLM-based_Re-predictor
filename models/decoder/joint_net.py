import torch.nn as nn

class JointNet(nn.Module):
    def __init__(self, encoder_dim, predictor_dim, joint_dim, vocab_size):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
        self.predictor_proj = nn.Linear(predictor_dim, joint_dim)
        self.activation = nn.Tanh()
        self.output = nn.Linear(joint_dim, vocab_size)

    def forward(self, encoder_out, predictor_out):
        """
        encoder_out: (B, T, encoder_dim)
        predictor_out: (B, U, predictor_dim)
        output: (B, T, U, vocab_size)
        """
        # 차원 확장: broadcasting 준비
        encoder_out = self.encoder_proj(encoder_out).unsqueeze(2)  # (B, T, 1, joint_dim)
        predictor_out = self.predictor_proj(predictor_out).unsqueeze(1)  # (B, 1, U, joint_dim)

        joint = self.activation(encoder_out + predictor_out)  # (B, T, U, joint_dim)
        logits = self.output(joint)  # (B, T, U, vocab_size)
        return logits
