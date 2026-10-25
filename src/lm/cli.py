"""Interactive inspection and evaluation for a saved language-model run.

Run from the repository root with:
    python src/lm/cli.py load results/<run-name> [--checkpoint ffnn_seed13.pt]
"""

import argparse
import csv
import json
import math
import shlex
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm.data import BOS, EOS, PAD, UNK
from lm.neural import build_model
from lm.statistical.ngram import NgramLM
from lm.neural.trainer import get_device


def iter_sentences(path: Path, lowercase: bool, percent: float = 100):
    limit = None
    if percent < 100:
        with path.open(encoding="utf-8") as handle:
            total = sum(1 for line in handle if line.strip())
        limit = math.ceil(total * percent / 100)
    with path.open(encoding="utf-8") as handle:
        yielded = 0
        for line in handle:
            sentence = line.split()
            if sentence:
                yield [word.lower() for word in sentence] if lowercase else sentence
                yielded += 1
                if limit is not None and yielded >= limit:
                    break


class CountNGram:
    """Adapter for the shared unsmoothed NgramLM using the run's saved vocabulary."""

    def __init__(self, order: int, vocab: list[str], sentences: Iterable[list[str]]):
        self.order = order
        self.stoi = {word: index for index, word in enumerate(vocab)}
        self.itos = vocab
        unk_id = self.stoi[UNK]

        def encoded_sentences():
            for sentence_idx, sentence in enumerate(sentences, start=1):
                if sentence_idx % 1_000_000 == 0:
                    print(f"  encoded training sentences: {sentence_idx:,}", flush=True)
                yield [self.stoi.get(word, unk_id) for word in sentence]

        self.model = NgramLM(
            order=order,
            vocab_size=len(vocab),
            bos=self.stoi[BOS],
            eos=self.stoi[EOS],
            smoothing="mle",
        ).fit(encoded_sentences())
        self.counts = self.model.ngram_counts[order]

    def score(self, context: tuple[int, ...], target: int) -> tuple[float, int | None]:
        probability = self.model.prob(context + (target,))
        followers = self.model.cont_following[self.order].get(context, ())
        predicted = (
            max(followers, key=lambda token: self.counts[context + (token,)])
            if followers else None
        )
        return probability, predicted


class LanguageModelCLI:
    def __init__(self, model_dir: Path, checkpoint_name: str | None, train_path: Path | None, device_name: str):
        self.model_dir = model_dir.expanduser().resolve()
        self.config = yaml.safe_load((self.model_dir / "config.yaml").read_text(encoding="utf-8"))
        self.vocab = json.loads((self.model_dir / "vocab.json").read_text(encoding="utf-8"))
        self.stoi = {word: index for index, word in enumerate(self.vocab)}
        self.device = get_device(device_name)
        self.lowercase = self.config.get("data", {}).get("lowercase", True)
        self.train_percent = float(self.config.get("train_percent", 100))
        self.project_root = Path(__file__).resolve().parents[2]
        self.train_path = train_path.expanduser().resolve() if train_path else self._configured_train_path()
        self.ngram_models = {}
        self.model, self.model_name, self.checkpoint_path = self._load_neural_model(checkpoint_name)

    def _configured_train_path(self) -> Path:
        data = self.config["data"]
        root = Path(data.get("output", "data/processed")).expanduser()
        if not root.is_absolute():
            root = self.project_root / root
        return root / data["corpus"] / "train.txt"

    def _load_neural_model(self, checkpoint_name: str | None):
        checkpoints = sorted(self.model_dir.glob("*_seed*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No model checkpoints (*_seed*.pt) found in {self.model_dir}")
        if checkpoint_name:
            requested = Path(checkpoint_name)
            checkpoint = requested if requested.is_absolute() else self.model_dir / requested
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        elif len(checkpoints) == 1:
            checkpoint = checkpoints[0]
        elif sys.stdin.isatty():
            print("Available checkpoints:")
            for index, path in enumerate(checkpoints, start=1):
                print(f"  {index}. {path.name}")
            choice = input("Choose checkpoint number: ").strip()
            checkpoint = checkpoints[int(choice) - 1]
        else:
            names = ", ".join(path.name for path in checkpoints)
            raise ValueError(f"Choose a checkpoint with --checkpoint. Available: {names}")

        model_name = checkpoint.stem.rsplit("_seed", 1)[0]
        model_config = next(
            (item for item in self.config["models"] if item["name"] == model_name), None
        )
        if model_config is None:
            raise ValueError(f"No model configuration for checkpoint '{model_name}'")
        model = build_model(model_name, len(self.vocab), self.stoi[PAD], **model_config.get("params", {}))
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        return model, model_name, checkpoint

    def _ngram(self, order: int):
        if order not in self.ngram_models:
            if not self.train_path.is_file():
                raise FileNotFoundError(
                    f"Training split not found: {self.train_path}. Load with --train PATH to set it."
                )
            print(f"Building unsmoothed {order}-gram counts from {self.train_path}", flush=True)
            self.ngram_models[order] = CountNGram(
                order,
                self.vocab,
                iter_sentences(self.train_path, self.lowercase, self.train_percent),
            )
            print(f"  counted {len(self.ngram_models[order].counts):,} distinct n-grams", flush=True)
        return self.ngram_models[order]

    def show_vocab(self, limit: int = 10):
        words = [word for word in self.vocab if word not in {PAD, UNK, BOS, EOS}]
        print(f"Top {min(limit, len(words))} vocabulary entries (frequency order):")
        for rank, word in enumerate(words[:limit], start=1):
            print(f"  {rank:>2}. {word}")
        print(f"Vocabulary size: {len(self.vocab):,} including special tokens")

    def show_stats(self):
        stats_path = self.model_dir / "data_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        print(f"Model: {self.model_name}")
        print(f"Checkpoint: {self.checkpoint_path.name}")
        print(f"Vocabulary size: {len(self.vocab):,}")
        print(f"Training split: {self.train_path}")
        for key, value in stats.items():
            if "oov_rate" in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")

    def _map_word(self, word: str) -> int:
        return self.stoi.get(word, self.stoi[UNK])

    @torch.no_grad()
    def predict(self, text: str, limit: int = 5):
        words = text.split()
        if self.lowercase:
            words = [word.lower() for word in words]
        unknown = sum(word not in self.stoi for word in words)
        ids = [self.stoi[BOS]] + [self._map_word(word) for word in words]
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model(inputs)[0, -1].float()
        logits[self.stoi[PAD]] = -torch.inf
        logits[self.stoi[BOS]] = -torch.inf
        probabilities = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probabilities, min(limit, len(self.vocab)))
        if unknown:
            print(f"Prompt OOV words mapped to <unk>: {unknown}/{len(words)}")
        print("Next-word predictions:")
        for probability, index in zip(values.tolist(), indices.tolist()):
            print(f"  {self.vocab[index]:<20} {probability:.4f}")

    def _neural_batch_scores(self, contexts: list[tuple[int, ...]], targets: list[int]):
        inputs = torch.tensor(contexts, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        logits = self.model(inputs)[:, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        losses = -log_probs.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
        predictions = logits.argmax(dim=-1)
        return losses.cpu().tolist(), predictions.cpu().tolist()

    @staticmethod
    def _summarize_metrics(name: str, total: int, zero_prob: int, nll: float, correct: int, covered: int):
        perplexity = math.inf if zero_prob else math.exp(nll / total) if total else math.nan
        finite_ppl = math.exp(nll / (total - zero_prob)) if total > zero_prob else math.inf
        print(f"{name}:")
        print(f"  perplexity: {perplexity:.4f}" if math.isfinite(perplexity) else "  perplexity: infinity")
        print(f"  finite-event perplexity: {finite_ppl:.4f}" if math.isfinite(finite_ppl) else "  finite-event perplexity: unavailable")
        print(f"  top-1 accuracy: {correct / covered:.2%}" if covered else "  top-1 accuracy: unavailable (no seen contexts)")
        print(f"  events with zero probability: {zero_prob:,}/{total:,} ({zero_prob / total:.2%})" if total else "  events: 0")

    def evaluate_file(self, path: Path, order: int):
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation file not found: {path}")
        ngram = self._ngram(order)
        bos_id, eos_id, unk_id = self.stoi[BOS], self.stoi[EOS], self.stoi[UNK]
        total = oov = word_total = ngram_zero = ngram_nll = ngram_correct = ngram_covered = 0
        neural_nll = neural_correct = 0.0
        neural_zero = 0
        context_batch, target_batch = [], []
        line_count = 0

        def flush_batch():
            nonlocal neural_nll, neural_correct
            if not context_batch:
                return
            losses, predictions = self._neural_batch_scores(context_batch, target_batch)
            neural_nll += sum(losses)
            neural_correct += sum(prediction == target for prediction, target in zip(predictions, target_batch))
            context_batch.clear()
            target_batch.clear()

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                words = line.split()
                if not words:
                    continue
                line_count += 1
                if self.lowercase:
                    words = [word.lower() for word in words]
                word_total += len(words)
                oov += sum(word not in self.stoi for word in words)
                ids = [self.stoi.get(word, unk_id) for word in words]
                sequence = [bos_id] * (order - 1) + ids + [eos_id]
                for end in range(order - 1, len(sequence)):
                    context = tuple(sequence[end - order + 1 : end])
                    target = sequence[end]
                    probability, predicted = ngram.score(context, target)
                    total += 1
                    if probability:
                        ngram_nll -= math.log(probability)
                    else:
                        ngram_zero += 1
                    if predicted is not None:
                        ngram_covered += 1
                        ngram_correct += predicted == target
                    context_batch.append(context)
                    target_batch.append(target)
                    if len(context_batch) >= 256:
                        flush_batch()
        flush_batch()

        print(f"Evaluation file: {path}")
        print(f"N-gram order: {order} ({'bigram' if order == 2 else 'trigram'})")
        print(f"Sentences: {line_count:,}; predicted tokens incl. </s>: {total:,}")
        print(f"OOV words: {oov:,}/{word_total:,} ({oov / word_total:.2%})" if word_total else "OOV words: 0")
        self._summarize_metrics("Unsmoothed count n-gram", total, ngram_zero, ngram_nll, ngram_correct, ngram_covered)
        self._summarize_metrics("Loaded neural model (same context window)", total, neural_zero, neural_nll, int(neural_correct), total)

    @staticmethod
    def _parse_row_range(value: str | None, row_count: int) -> tuple[int, int]:
        """Parse a zero-based, end-exclusive range such as 0-10 or 10-all."""
        if value is None:
            return 0, min(10, row_count)
        try:
            start_text, end_text = value.strip().lower().split("-", 1)
            start = int(start_text)
            end_text = end_text.strip().strip("() ")
            end = row_count if end_text in {"all", "*"} else int(end_text)
        except ValueError as exc:
            raise ValueError("range must look like START-END, for example 0-10 or 10-all") from exc
        if start < 0 or end < 0 or end < start:
            raise ValueError("range indexes must be non-negative and END must be >= START")
        return min(start, row_count), min(end, row_count)

    def rare_check(self, path: Path, order: int, row_range: str | None = None):
        if not path.is_file():
            raise FileNotFoundError(f"Rare-sequence CSV not found: {path}")
        records = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    row_order = int(row.get("order", order))
                except (ValueError, TypeError):
                    continue
                sequence = (row.get("sequence") or "").split()
                if row_order == order and len(sequence) == order:
                    if self.lowercase:
                        sequence = [word.lower() for word in sequence]
                    try:
                        weight = max(1, int(row.get("test_count", 1)))
                    except (ValueError, TypeError):
                        weight = 1
                    records.append((sequence, weight))
        if not records:
            print(f"No order-{order} sequences found in {path}")
            return

        start, end = self._parse_row_range(row_range, len(records))
        records = records[start:end]
        if not records:
            print(f"No order-{order} sequences in range {start}-{end} from {path}")
            return

        weighted_total = 0
        neural_nll = 0.0
        contexts, targets, weights = [], [], []
        for words, weight in records:
            context_words, target_word = words[:-1], words[-1]
            context = tuple(self._map_word(word) for word in context_words)
            target = self._map_word(target_word)
            weighted_total += weight
            contexts.append(context)
            targets.append(target)
            weights.append(weight)

        neural_correct = 0
        neural_predictions = []
        for start in range(0, len(contexts), 256):
            batch_contexts = contexts[start : start + 256]
            batch_targets = targets[start : start + 256]
            batch_weights = weights[start : start + 256]
            losses, predictions = self._neural_batch_scores(batch_contexts, batch_targets)
            neural_nll += sum(loss * weight for loss, weight in zip(losses, batch_weights))
            neural_correct += sum(weight for pred, target, weight in zip(predictions, batch_targets, batch_weights) if pred == target)
            neural_predictions.extend(zip(predictions, losses))

        print(
            f"Rare CSV: {path}; selected {order}-gram rows {start}-{end} "
            f"({len(records):,} rows)"
        )
        print(f"Selected model: {self.model_name} ({self.checkpoint_path.name})")
        print(f"CSV test-event weight: {weighted_total:,}")
        self._summarize_metrics(
            "Selected model (test_count weighted)",
            weighted_total,
            0,
            neural_nll,
            neural_correct,
            weighted_total,
        )
        print("Examples (context | actual next token | selected-model prediction | test_count):")
        for idx, (words, weight) in enumerate(records):
            neural_id, _ = neural_predictions[idx]
            neural_word = self.vocab[neural_id]
            print(
                f"  {' '.join(words[:-1])} | {words[-1]} | {neural_word} | test_count={weight}"
            )

    def help(self):
        print("Commands:")
        print("  vocab [N]                 show the top N words (default 10)")
        print("  stats                     show vocabulary and OOV/split statistics")
        print("  predict TEXT              show the loaded neural model's next-word guesses")
        print("  evaluate FILE bi|tri      evaluate count and neural models on a text file")
        print("  rare-check bi|tri CSV [RANGE]  score CSV rows (default 0-10; use 0-all for all)")
        print("  help | quit")

    def shell(self):
        print(f"Loaded {self.model_name}: {self.checkpoint_path.name} on {self.device}")
        print(f"Training data for n-gram evaluation: {self.train_path}")
        self.help()
        while True:
            try:
                raw = input("lm> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            stripped = raw.strip()
            if stripped.lower().startswith("predict "):
                self.predict(stripped[len("predict ") :])
                continue
            try:
                parts = shlex.split(raw)
                if not parts:
                    continue
                command = parts[0].lower()
                if command in {"quit", "exit", "q"}:
                    break
                if command == "help":
                    self.help()
                elif command == "vocab":
                    self.show_vocab(int(parts[1]) if len(parts) > 1 else 10)
                elif command == "stats":
                    self.show_stats()
                elif command == "predict":
                    if len(parts) < 2:
                        print("Usage: predict TEXT")
                    else:
                        self.predict(" ".join(parts[1:]))
                elif command == "evaluate" and len(parts) == 3:
                    order = self._parse_order(parts[2])
                    self.evaluate_file(Path(parts[1]).expanduser(), order)
                elif command == "rare-check" and len(parts) in {3, 4}:
                    order = self._parse_order(parts[1])
                    self.rare_check(
                        Path(parts[2]).expanduser(),
                        order,
                        parts[3] if len(parts) == 4 else None,
                    )
                else:
                    # Treat a plain line as a next-word prompt for quick interactive use.
                    self.predict(raw)
            except ValueError as exc:
                if "quotation" in str(exc).lower():
                    self.predict(raw)
                else:
                    print(f"error: {exc}")
            except (FileNotFoundError, IndexError) as exc:
                print(f"error: {exc}")

    @staticmethod
    def _parse_order(value: str) -> int:
        normalized = value.lower()
        if normalized in {"bi", "bigram", "2"}:
            return 2
        if normalized in {"tri", "trigram", "3"}:
            return 3
        raise ValueError("order must be bi or tri")


def main():
    parser = argparse.ArgumentParser(description="Inspect and evaluate a saved language-model run.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    load_parser = subparsers.add_parser("load", help="load a trained model folder and open the CLI")
    load_parser.add_argument("model_folder", type=Path)
    load_parser.add_argument("--checkpoint", help="checkpoint filename, e.g. ffnn_seed13.pt")
    load_parser.add_argument("--train", type=Path, help="override the training split path")
    load_parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()
    cli = LanguageModelCLI(args.model_folder, args.checkpoint, args.train, args.device)
    cli.shell()


if __name__ == "__main__":
    main()
