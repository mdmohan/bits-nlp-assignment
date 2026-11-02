"""Model-agnostic training and evaluation for next-word prediction.

Works with any model that follows the contract in lm/neural/__init__.py and any DataLoader that yields
(inputs, targets) id tensors of shape (batch, seq_len), with padding marked by `pad_id`.
"""

import copy
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data import mask_logits


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
    def __init__(self, model: nn.Module, config: TrainConfig, pad_id: int, device: torch.device,
                 allowed_target_ids=None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        # Sum over tokens so perplexity is total NLL / total tokens, never an average of averages.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
        self.pad_id = pad_id
        self.allowed_target_ids = allowed_target_ids
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def _batch_loss(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = mask_logits(self.model(inputs), self.allowed_target_ids)
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

    @staticmethod
    def _capture_rng_state() -> dict:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "get_rng_state"):
            try:
                state["torch_mps"] = mps.get_rng_state()
            except (RuntimeError, TypeError):
                pass
        return state

    @staticmethod
    def _restore_rng_state(state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        if "torch_mps" in state:
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "set_rng_state"):
                mps.set_rng_state(state["torch_mps"])

    @staticmethod
    def _atomic_torch_save(value, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(value, temporary)
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def train_epoch(
        self,
        loader,
        progress_label: str = "train",
    ) -> dict:
        """Train one complete epoch."""
        self.model.train()
        total_batches = len(loader)
        started_at = time.perf_counter()
        total_nll = 0.0
        total_tokens = 0
        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            nll, num_tokens = self._batch_loss(inputs, targets)
            self.optimizer.zero_grad()
            (nll / num_tokens).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total_nll += nll.item()
            total_tokens += num_tokens
            self._log_batch_progress(
                progress_label, batch_idx, total_batches, started_at, total_nll, total_tokens,
            )
        return {
            "perplexity": math.exp(total_nll / total_tokens),
            "nll_sum": total_nll,
            "tokens": total_tokens,
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        progress_label: str = "eval",
    ) -> dict:
        """Corpus perplexity = exp(sum of token NLL / number of predicted tokens, incl. </s>)."""
        self.model.eval()
        total_batches = len(loader)
        started_at = time.perf_counter()
        total_nll = 0.0
        total_tokens = 0
        for batch_idx, (inputs, targets) in enumerate(loader, start=1):
            nll, num_tokens = self._batch_loss(inputs, targets)
            total_nll += nll.item()
            total_tokens += num_tokens
            self._log_batch_progress(
                progress_label, batch_idx, total_batches, started_at, total_nll, total_tokens,
            )
        return {
            "nll": total_nll / total_tokens,
            "perplexity": math.exp(total_nll / total_tokens),
            "tokens": total_tokens,
            "nll_sum": total_nll,
        }

    def fit(
        self,
        train_loader,
        valid_loader,
        epoch_callback=None,
        checkpoint_path: str | Path | None = None,
        best_checkpoint_path: str | Path | None = None,
        checkpoint_metadata: dict | None = None,
        resume: bool = False,
    ) -> list[dict]:
        """Fit with an atomic checkpoint after each completed train/validation epoch."""
        checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        best_checkpoint_path = (
            Path(best_checkpoint_path) if best_checkpoint_path is not None
            else checkpoint_path.with_name(checkpoint_path.stem + ".best.pt")
            if checkpoint_path is not None else None
        )
        best_state_path = None
        best_state_in_memory = None
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if resume and checkpoint_path.is_file():
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if saved.get("format_version") != 2:
                    raise ValueError(f"Unsupported training checkpoint format: {checkpoint_path}")
                if saved.get("metadata") != (checkpoint_metadata or {}):
                    raise ValueError(
                        "Checkpoint settings or input data do not match this run. "
                        "Use the original config/data or start a fresh run without --resume."
                    )
                self.model.load_state_dict(saved["model_state"])
                self.optimizer.load_state_dict(saved["optimizer_state"])
                history = list(saved["history"])
                best_ppl = saved["best_ppl"]
                best_epoch = saved["best_epoch"]
                epochs_without_improvement = saved["epochs_without_improvement"]
                completed_epoch = int(saved["completed_epoch"])
                stopped_early = bool(saved.get("stopped_early", False))
                best_state_path = Path(saved["best_checkpoint_file"])
                if not best_state_path.is_file():
                    raise FileNotFoundError(
                        f"Best-model checkpoint referenced by {checkpoint_path} is missing: "
                        f"{best_state_path}"
                    )
                if best_checkpoint_path is not None:
                    for stale_best in best_checkpoint_path.parent.glob(
                        f"{best_checkpoint_path.stem}.epoch-*.pt"
                    ):
                        if stale_best != best_state_path:
                            stale_best.unlink(missing_ok=True)
                rng_state = saved["rng_state"]
                self._restore_rng_state(rng_state)
                print(
                    f"  Resuming after completed epoch {completed_epoch}; "
                    f"next epoch={completed_epoch + 1}", flush=True,
                )
            else:
                checkpoint_path.unlink(missing_ok=True)
                if best_checkpoint_path is not None:
                    best_checkpoint_path.unlink(missing_ok=True)
                    for stale_best in best_checkpoint_path.parent.glob(
                        f"{best_checkpoint_path.stem}.epoch-*.pt"
                    ):
                        stale_best.unlink(missing_ok=True)
                checkpoint_path.with_name(checkpoint_path.name + ".tmp").unlink(missing_ok=True)
                history = []
                best_ppl, best_epoch, epochs_without_improvement = math.inf, 0, 0
                completed_epoch = 0
                stopped_early = False
        else:
            history = []
            best_ppl, best_epoch, epochs_without_improvement = math.inf, 0, 0
            completed_epoch = 0
            stopped_early = False

        epochs_to_run = () if stopped_early else range(completed_epoch + 1, self.config.epochs + 1)
        for epoch in epochs_to_run:
            epoch_started = time.perf_counter()
            train_result = self.train_epoch(
                train_loader, f"train epoch {epoch}/{self.config.epochs}"
            )
            train_ppl = train_result["perplexity"]
            valid_result = self.evaluate(
                valid_loader, f"valid epoch {epoch}/{self.config.epochs}"
            )
            valid_ppl = valid_result["perplexity"]
            seconds = time.perf_counter() - epoch_started
            history.append({
                "epoch": epoch, "train_ppl": train_ppl,
                "valid_ppl": valid_ppl, "seconds": seconds,
            })
            print(
                f"  epoch {epoch:2d} | train ppl {train_ppl:8.2f} "
                f"| valid ppl {valid_ppl:8.2f} | {seconds:.1f}s",
                flush=True,
            )

            if best_epoch == 0 or valid_ppl < best_ppl:
                best_ppl, best_epoch, epochs_without_improvement = valid_ppl, epoch, 0
                if best_checkpoint_path is not None:
                    best_state_path = best_checkpoint_path.with_name(
                        f"{best_checkpoint_path.stem}.epoch-{epoch}.pt"
                    )
                    self._atomic_torch_save(self.model.state_dict(), best_state_path)
                else:
                    best_state_in_memory = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1
            if epoch_callback is not None:
                epoch_callback(history[-1], self.model)

            stop_early = epochs_without_improvement >= self.config.patience
            if stop_early:
                print(f"  early stopping: no improvement for {self.config.patience} epochs")
            completed_epoch = epoch
            stopped_early = stop_early
            if checkpoint_path is not None:
                state = {
                    "format_version": 2,
                    "metadata": checkpoint_metadata or {},
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "history": history,
                    "best_ppl": best_ppl,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "completed_epoch": completed_epoch,
                    "stopped_early": stopped_early,
                    "best_checkpoint_file": str(best_state_path),
                    "rng_state": self._capture_rng_state(),
                }
                self._atomic_torch_save(state, checkpoint_path)
                # Only remove older best weights after the latest checkpoint
                # atomically points at the newly selected best epoch.
                for stale_best in best_checkpoint_path.parent.glob(
                    f"{best_checkpoint_path.stem}.epoch-*.pt"
                ):
                    if best_state_path is None or stale_best != best_state_path:
                        stale_best.unlink(missing_ok=True)
            if stop_early:
                break

        if best_state_path is not None and best_state_path.is_file():
            best_state = torch.load(best_state_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(best_state)
        elif best_state_in_memory is not None:
            self.model.load_state_dict(best_state_in_memory)
        print(f"  best validation epoch: {best_epoch} (ppl {best_ppl:.2f})", flush=True)
        return history
