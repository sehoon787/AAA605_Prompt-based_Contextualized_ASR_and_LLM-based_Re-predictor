import torch
import torch.nn as nn
from models.decoder.stateless_prediction_net import StatelessPredictionNet
from models.decoder.joint_net import JointNet

class ASRDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prediction_net = StatelessPredictionNet(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
        )

        self.joint_net = JointNet(
            encoder_dim=config["encoder_output_dim"],
            predictor_dim=config["embed_dim"],  # embed_dim == predictor_dim
            joint_dim=config["joint_dim"],
            vocab_size=config["vocab_size"]
        )

    def forward(self, encoder_out, tokens):
        """
        encoder_out: (B, T, encoder_output_dim)
        tokens: (B, U)  -> target tokens for prediction network
        """
        predictor_out = self.prediction_net(tokens)  # (B, U, embed_dim)
        print("StatelessPredictionNet output NaN:", torch.isnan(predictor_out).any())

        # blank token prepending
        blank = torch.zeros((predictor_out.size(0), 1, predictor_out.size(2)),
                             device=predictor_out.device, dtype=predictor_out.dtype)
        predictor_out = torch.cat([blank, predictor_out], dim=1)  # (B, U+1, embed_dim)

        logits = self.joint_net(encoder_out, predictor_out)  # (B, T, U+1, V)
        print("JointNet output NaN:", torch.isnan(logits).any())
        return logits
