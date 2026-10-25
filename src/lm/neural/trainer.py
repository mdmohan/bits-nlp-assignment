"""Model-agnostic training and evaluation for next-word prediction.

Works with any model that follows the contract in lm/neural/__init__.py and any DataLoader that yields
(inputs, targets) id tensors of shape (batch, seq_len), with padding marked by `pad_id`.
"""

import copy
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    patience: int = 3  # stop after this many epochs without validation improvement


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig, pad_id: int, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        # Sum over tokens so perplexity is total NLL / total tokens, never an average of averages.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
        self.pad_id = pad_id
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def _batch_loss(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = self.model(inputs)
        total_nll = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        num_tokens = (targets != self.pad_id).sum().item()
        return total_nll, num_tokens

    @staticmethod
    def _log_batch_progress(phase, batch_idx, total_batches, started_at, total_nll, total_tokens):
        # Keep logs readable on large corpora while still showing regular progress.
        update_every = max(1, math.ceil(total_batches / 20))
        if batch_idx != 1 and batch_idx != total_batches and batch_idx % update_every:
            return

        elapsed = time.perf_counter() - started_at
        eta = elapsed / batch_idx * (total_batches - batch_idx)
        running_ppl = math.exp(total_nll / total_tokens) if total_tokens else float("nan")
        percent = 100 * batch_idx / total_batches
        print(
            f"    {phase}: batch {batch_idx:,}/{total_batches:,} ({percent:5.1f}%) "
            f"running ppl {running_ppl:8.2f} | elapsed {elapsed:.1f}s | eta {eta:.1f}s",
            flush=True,
        )

    def train_epoch(self, loader, progress_label: str = "train") -> float:
        """One pass over the training data; returns the training perplexity."""
        self.model.train()
        total_nll, total_tokens = 0.0, 0
        total_batches = len(loader)
        started_at = time.perf_counter()
        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            nll, num_tokens = self._batch_loss(inputs, targets)
            self.optimizer.zero_grad()
            (nll / num_tokens).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total_nll += nll.item()
            total_tokens += num_tokens
            self._log_batch_progress(
                progress_label, batch_idx, total_batches, started_at, total_nll, total_tokens
            )
        return math.exp(total_nll / total_tokens)

    @torch.no_grad()
    def evaluate(self, loader, progress_label: str = "eval") -> dict:
        """Corpus perplexity = exp(sum of token NLL / number of predicted tokens, incl. </s>)."""
        self.model.eval()
        total_nll, total_tokens = 0.0, 0
        total_batches = len(loader)
        started_at = time.perf_counter()
        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            nll, num_tokens = self._batch_loss(inputs, targets)
            total_nll += nll.item()
            total_tokens += num_tokens
            self._log_batch_progress(
                progress_label, batch_idx, total_batches, started_at, total_nll, total_tokens
            )
        return {
            "nll": total_nll / total_tokens,
            "perplexity": math.exp(total_nll / total_tokens),
            "tokens": total_tokens,
        }

    def fit(self, train_loader, valid_loader) -> list[dict]:
        """Train with early stopping on validation perplexity; keeps the best weights."""
        history = []
        best_ppl, best_state, epochs_without_improvement = math.inf, None, 0

        for epoch in range(1, self.config.epochs + 1):
            start = time.perf_counter()
            train_ppl = self.train_epoch(train_loader, f"train epoch {epoch}/{self.config.epochs}")
            valid_ppl = self.evaluate(
                valid_loader, f"valid epoch {epoch}/{self.config.epochs}"
            )["perplexity"]
            seconds = time.perf_counter() - start

            history.append({"epoch": epoch, "train_ppl": train_ppl, "valid_ppl": valid_ppl, "seconds": seconds})
            print(f"  epoch {epoch:2d} | train ppl {train_ppl:8.2f} | valid ppl {valid_ppl:8.2f} | {seconds:.1f}s")

            if valid_ppl < best_ppl:
                best_ppl, epochs_without_improvement = valid_ppl, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    print(f"  early stopping: no improvement for {self.config.patience} epochs")
                    break

        self.model.load_state_dict(best_state)
        return history
