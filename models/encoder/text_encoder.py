import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state  # (B, L, H)
