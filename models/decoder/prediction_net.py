import torch
import torch.nn as nn

class PredictionNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, tokens, hidden=None):
        embedded = self.embedding(tokens)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden
