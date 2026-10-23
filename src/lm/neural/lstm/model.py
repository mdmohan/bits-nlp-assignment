from torch import nn


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
