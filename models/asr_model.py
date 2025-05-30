import torch.nn as nn
from models.encoder.asr_encoder import ASREncoder
from models.decoder.asr_decoder import ASRDecoder

class ASRModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = ASREncoder(config)
        self.decoder = ASRDecoder(config)

    def forward(self, speech_input, input_ids, attention_mask, tokens):
        encoder_out = self.encoder(speech_input, input_ids, attention_mask)
        logits = self.decoder(encoder_out, tokens)
        return logits
