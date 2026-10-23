"""Neural language models.

Model contract (what the trainer and controller rely on):
    model = ModelClass(vocab_size, pad_id, **params)
    logits = model(inputs)   # inputs: (batch, seq_len) word ids -> logits: (batch, seq_len, vocab_size)
logits[:, t] is the prediction for the word that follows inputs[:, t].

To add a model, create a subpackage (like ffnn/ or lstm/) with a class that follows this
contract and add it to MODELS.
"""

from torch import nn

from .ffnn import FeedForwardLM
from .lstm import LSTMLM

MODELS = {
    "ffnn": FeedForwardLM,
    "lstm": LSTMLM,
}


def build_model(name: str, vocab_size: int, pad_id: int, **params) -> nn.Module:
    return MODELS[name](vocab_size, pad_id, **params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
