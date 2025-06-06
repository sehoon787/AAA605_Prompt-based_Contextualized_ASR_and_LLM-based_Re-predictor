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

    def forward(self, speech_input, input_ids, attention_mask):
        speech_features = self.speech_encoder(speech_input)
        # print("speech_features:", speech_features.shape)
        text_features = self.text_encoder(input_ids, attention_mask)
        # print("text_features:", text_features.shape)
        text_adapted = self.text_adapter(text_features)
        # print("text_adapted:", text_adapted.shape)
        fused_output = self.cross_attention(speech_features, text_adapted)
        # print("fused_output:", fused_output.shape)
        return fused_output
