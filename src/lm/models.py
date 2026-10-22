"""Neural language models.

Model contract (what the trainer and controller rely on):
    model = ModelClass(vocab_size, pad_id, **params)
    logits = model(inputs)   # inputs: (batch, seq_len) word ids -> logits: (batch, seq_len, vocab_size)
logits[:, t] is the prediction for the word that follows inputs[:, t].

To add a model, write a class with that contract and add it to MODELS.
"""

import torch
from torch import nn


class FeedForwardLM(nn.Module):
    """Fixed-window neural LM (Bengio et al., 2003): the previous `context_size` words are
    embedded, concatenated and passed through one hidden layer."""

    def __init__(self, vocab_size, pad_id, context_size=3, embed_dim=128, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.context_size = context_size
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.hidden = nn.Sequential(
            nn.Linear(context_size * embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        # Left-pad so the first positions also have a full window (padding embeds to zero).
        padding = inputs.new_full((batch_size, self.context_size - 1), self.pad_id)
        padded = torch.cat([padding, inputs], dim=1)
        windows = padded.unfold(dimension=1, size=self.context_size, step=1)  # (B, T, context)
        features = self.embedding(windows).flatten(start_dim=2)               # (B, T, context*embed)
        return self.output(self.hidden(features))


class LSTMLM(nn.Module):
    """Word-level LSTM LM: each prediction can use the whole sentence prefix."""

    def __init__(self, vocab_size, pad_id, embed_dim=128, hidden_dim=256, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embedded = self.dropout(self.embedding(inputs))
        hidden_states, _ = self.lstm(embedded)
        return self.output(self.dropout(hidden_states))


MODELS = {
    "ffnn": FeedForwardLM,
    "lstm": LSTMLM,
}


def build_model(name: str, vocab_size: int, pad_id: int, **params) -> nn.Module:
    return MODELS[name](vocab_size, pad_id, **params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
