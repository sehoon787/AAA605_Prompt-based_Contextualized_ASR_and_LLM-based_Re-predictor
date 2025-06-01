import torch.nn as nn

class StatelessPredictionNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        projected = self.projection(embedded)
        return projected
