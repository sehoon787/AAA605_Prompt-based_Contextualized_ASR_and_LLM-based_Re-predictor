import torch
import torch.nn as nn

from models.encoder.speech_encoder import SpeechEncoder
from models.encoder.text_encoder import TextEncoder
from models.encoder.text_adapter import TextAdapter
from models.encoder.cross_attention import CrossAttentionFusion

class ASREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.speech_encoder = SpeechEncoder(
            input_dim=config["speech_input_dim"],
            hidden_dim=config["speech_hidden_dim"],
            zipformer_blocks=config["zipformer_blocks"],
            reduction_factors=config["reduction_factors"],
        )

        self.text_encoder = TextEncoder(
            pretrained_model_name=config["pretrained_model_name"]
        )

        self.text_adapter = TextAdapter(
            input_dim=config["bert_hidden_dim"],
            adapter_dim=config["adapter_dim"]
        )

        self.cross_attention = CrossAttentionFusion(
            audio_dim=config["speech_hidden_dim"],
            text_dim=config["adapter_dim"],
            fusion_dim=config["fusion_dim"]
        )

    def forward(self, speech_input, utterance_ids, utterance_mask):
        speech_features = self.speech_encoder(speech_input)
        text_features = self.text_encoder(utterance_ids, utterance_mask)
        text_adapted = self.text_adapter(text_features)
        fused_output = self.cross_attention(speech_features, text_adapted)

        print("SpeechEncoder output NaN:", torch.isnan(speech_features).any())
        print("TextEncoder output NaN:", torch.isnan(text_features).any())
        print("TextAdapter output NaN:", torch.isnan(text_adapted).any())
        print("CrossAttention output NaN:", torch.isnan(fused_output).any())

        return fused_output
