"""Small LSTM language model, trained from scratch.

NOTE ON PLACEMENT
-----------------
This is a standalone reference implementation kept alongside ``ngram.py`` so
that the PS-1 count-based vs neural comparison lives in one place and can be
run with no framework dependencies. It is INDEPENDENT of the package's main
neural implementation under ``src/lm/neural/``, which integrates with
``controller.py`` and ``trainer.py``. Use ``src/lm/neural/`` for experiments
driven by the shared controller; use this file for the self-contained
comparison reported in the PS-1 write-up.

Self-contained apart from numpy and torch: no project-local imports, so this
file can be dropped into any project layout.

No pretrained weights are used. That keeps the comparison against the
count-based models in ``ngram.py`` fair -- both see exactly the same training
tokens and the same vocabulary -- and avoids any pretrained-model licensing
declaration.

Modelling choice worth defending in the paper: the model is trained and scored
SENTENCE BY SENTENCE, with the hidden state reset at every sentence boundary.
Carrying state across sentences would give the neural model context the n-gram
models cannot see, making the comparison unfair.

Token accounting matches ``ngram.py`` exactly: a sentence ``w1 ... wT`` yields
``T + 1`` predictions.

    input   <s> w1 w2 ... wT
    target  w1  w2 ... wT </s>

Usage
-----
    from neural_lm import train_lstm, NeuralScorer

    model, meta = train_lstm(train_enc, valid_enc, vocab_size=len(vocab),
                             bos=BOS_ID, eos=EOS_ID, pad_input=EOS_ID, seed=42)

    scorer = NeuralScorer(model, BOS_ID, EOS_ID, pad_input=EOS_ID)
    scorer.token_logprobs(encoded_sentence)        # one sentence
    scorer.batch_token_logprobs(encoded_corpus)    # batched, much faster
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ["LSTMConfig", "LSTMLanguageModel", "NeuralScorer", "train_lstm",
           "set_seed", "make_batch", "iter_batches", "IGNORE"]

IGNORE = -100  # CrossEntropyLoss ignore_index for padded targets


@dataclass
class LSTMConfig:
    """Architecture and optimisation settings.

    Defaults match the configuration reported in the paper. Pass an instance
    to ``train_lstm`` to override any of them.
    """

    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    batch_size: int = 32
    learning_rate: float = 1e-3
    clip_norm: float = 5.0
    max_epochs: int = 30
    patience: int = 3  # early stopping on validation perplexity


DEFAULT_CONFIG = LSTMConfig()


def set_seed(seed: int) -> None:
    """Seed every source of randomness used during training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------
def make_batch(sentences: list[list[int]], bos: int, eos: int, pad_input: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (inputs, targets, lengths) for a list of encoded sentences.

    Padded target positions are set to ``IGNORE`` so they contribute nothing
    to the loss. ``pad_input`` may be any valid id; padded inputs never reach
    the LSTM because the sequence is packed.
    """
    lengths = [len(s) + 1 for s in sentences]
    width = max(lengths)
    inp = torch.full((len(sentences), width), pad_input, dtype=torch.long)
    tgt = torch.full((len(sentences), width), IGNORE, dtype=torch.long)
    for i, s in enumerate(sentences):
        seq_in = [bos] + list(s)
        seq_out = list(s) + [eos]
        inp[i, :len(seq_in)] = torch.tensor(seq_in, dtype=torch.long)
        tgt[i, :len(seq_out)] = torch.tensor(seq_out, dtype=torch.long)
    return inp, tgt, torch.tensor(lengths, dtype=torch.long)


def iter_batches(sentences: list[list[int]], batch_size: int, shuffle: bool,
                 rng: random.Random | None = None):
    """Yield (original indices, sentences) per batch.

    Evaluation batches are length-sorted to cut padding waste; training
    batches are shuffled with a seeded generator for reproducibility.
    """
    idx = list(range(len(sentences)))
    if shuffle:
        (rng or random).shuffle(idx)
    else:
        idx.sort(key=lambda i: len(sentences[i]))
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        yield chunk, [sentences[i] for i in chunk]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class LSTMLanguageModel(nn.Module):
    """Embedding -> LSTM -> linear projection over the vocabulary."""

    def __init__(self, vocab_size: int, cfg: LSTMConfig = DEFAULT_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.embed_dim)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.hidden_dim,
                            num_layers=cfg.num_layers,
                            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                            batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.hidden_dim, vocab_size)

    def forward(self, inp: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.drop(self.embed(inp))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=inp.size(1))
        return self.out(self.drop(out))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Scorer -- same interface as NgramLM.token_logprobs
# --------------------------------------------------------------------------
class NeuralScorer:
    """Wraps a trained model so it can be scored exactly like an n-gram LM."""

    def __init__(self, model: LSTMLanguageModel, bos: int, eos: int,
                 pad_input: int, name: str = "LSTM", batch_size: int = 32):
        self.model = model
        self.bos = bos
        self.eos = eos
        self.pad_input = pad_input
        self.name = name
        self.batch_size = batch_size

    @torch.no_grad()
    def batch_token_logprobs(self, encoded: list[list[int]]) -> list[list[float]]:
        """Per-prediction log probabilities for a whole corpus, batched."""
        self.model.eval()
        out: list[list[float]] = [[] for _ in encoded]
        for chunk, sents in iter_batches(encoded, self.batch_size, shuffle=False):
            inp, tgt, lengths = make_batch(sents, self.bos, self.eos, self.pad_input)
            logits = self.model(inp, lengths)
            logp = torch.log_softmax(logits, dim=-1)
            safe = tgt.clone()
            safe[safe == IGNORE] = 0
            picked = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            for row, orig_i in enumerate(chunk):
                n = lengths[row].item()
                out[orig_i] = picked[row, :n].tolist()
        return out

    def token_logprobs(self, ids: list[int]) -> list[float]:
        """Log probability of each prediction, aligned with [w1 ... wT, </s>]."""
        return self.batch_token_logprobs([ids])[0]

    def sentence_logprob(self, ids: list[int]) -> tuple[float, int]:
        """(natural-log probability of the sentence, number of predictions)."""
        lps = self.token_logprobs(ids)
        return sum(lps), len(lps)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_lstm(train: list[list[int]], valid: list[list[int]], vocab_size: int,
               bos: int, eos: int, pad_input: int, seed: int,
               cfg: LSTMConfig = DEFAULT_CONFIG,
               verbose: bool = True) -> tuple[LSTMLanguageModel, dict]:
    """Train with early stopping on validation perplexity.

    Returns the best model (weights restored from the best epoch) and a
    metadata dict containing the per-epoch history, parameter count and
    wall-clock training time.
    """
    set_seed(seed)
    rng = random.Random(seed)

    model = LSTMLanguageModel(vocab_size, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")

    best_ppl = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    t0 = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tot_loss = tot_tok = 0
        for _, sents in iter_batches(train, cfg.batch_size, shuffle=True, rng=rng):
            inp, tgt, lengths = make_batch(sents, bos, eos, pad_input)
            opt.zero_grad()
            logits = model(inp, lengths)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            n_tok = int((tgt != IGNORE).sum())
            (loss / n_tok).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step()
            tot_loss += float(loss.detach())
            tot_tok += n_tok

        train_ppl = math.exp(tot_loss / tot_tok)
        val_ppl = evaluate_perplexity(model, valid, bos, eos, pad_input, cfg)
        history.append({"epoch": epoch, "train_ppl": train_ppl, "valid_ppl": val_ppl})
        if verbose:
            print(f"    epoch {epoch:2d}  train_ppl={train_ppl:8.2f}  "
                  f"valid_ppl={val_ppl:8.2f}")

        if val_ppl < best_ppl - 1e-4:
            best_ppl = val_ppl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"    early stop at epoch {epoch} "
                          f"(best valid_ppl={best_ppl:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "seed": seed,
        "best_valid_ppl": best_ppl,
        "epochs_run": len(history),
        "train_seconds": time.perf_counter() - t0,
        "n_parameters": model.n_parameters(),
        "history": history,
    }
    return model, meta


@torch.no_grad()
def evaluate_perplexity(model: LSTMLanguageModel, data: list[list[int]],
                        bos: int, eos: int, pad_input: int,
                        cfg: LSTMConfig = DEFAULT_CONFIG) -> float:
    """Corpus perplexity, base e, averaged per predicted token."""
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")
    tot_loss = tot_tok = 0
    for _, sents in iter_batches(data, cfg.batch_size, shuffle=False):
        inp, tgt, lengths = make_batch(sents, bos, eos, pad_input)
        logits = model(inp, lengths)
        tot_loss += float(crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)))
        tot_tok += int((tgt != IGNORE).sum())
    return math.exp(tot_loss / tot_tok)
