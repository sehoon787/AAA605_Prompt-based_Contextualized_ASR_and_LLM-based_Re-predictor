import torch.nn as nn

class JointNet(nn.Module):
    def __init__(self, encoder_dim, predictor_dim, joint_dim, vocab_size):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
        self.predictor_proj = nn.Linear(predictor_dim, joint_dim)
        self.activation = nn.Tanh()

        self.output = nn.Linear(joint_dim, vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.encoder_proj.weight)
        nn.init.xavier_uniform_(self.predictor_proj.weight)
        nn.init.xavier_uniform_(self.output.weight, gain=0.1)
        nn.init.zeros_(self.output.bias)

    def forward(self, encoder_out, predictor_out):
        encoder_out = self.encoder_proj(encoder_out).unsqueeze(2)
        predictor_out = self.predictor_proj(predictor_out).unsqueeze(1)
        joint = self.activation(encoder_out + predictor_out)
        logits = self.output(joint)
        return logits
