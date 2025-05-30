import torch.nn as nn
from models.decoder.prediction_net import PredictionNet
from models.decoder.joint_net import JointNet

class ASRDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prediction_net = PredictionNet(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            hidden_dim=config["predictor_hidden_dim"]
        )

        self.joint_net = JointNet(
            encoder_dim=config["encoder_output_dim"],
            predictor_dim=config["predictor_hidden_dim"],
            joint_dim=config["joint_dim"],
            vocab_size=config["vocab_size"]
        )

    def forward(self, encoder_out, tokens):
        predictor_out, _ = self.prediction_net(tokens)
        logits = self.joint_net(encoder_out, predictor_out)
        return logits
