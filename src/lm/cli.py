"""Interactive shell for loading and evaluating language models."""

import argparse
import atexit
import csv
import glob
import heapq
import json
import math
import os
import re
import shlex
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch
import yaml

try:
    import readline
except ImportError:  # Windows and some minimal Python builds
    readline = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm.data import BOS, EOS, PAD, UNK, Vocab, mask_logits
from lm.evaluation import evaluate_model, sentence_events
from lm.neural import build_model
from lm.statistical.ngram import NgramLM
from lm.neural.trainer import get_device
from lm.tokenization import canonical_tokenizer_name, get_tokenizer, tokenize


def iter_sentences(path: Path, lowercase: bool, percent: float = 100,
                   tokenizer_name: str = "space"):
    tokenizer = get_tokenizer(tokenizer_name)
    limit = None
    if percent < 100:
        with path.open(encoding="utf-8") as handle:
            total = sum(1 for line in handle if line.strip())
        limit = math.ceil(total * percent / 100)
    with path.open(encoding="utf-8") as handle:
        yielded = 0
        for line in handle:
            sentence = tokenizer.tokenize(line)
            if sentence:
                yield [word.lower() for word in sentence] if lowercase else sentence
                yielded += 1
                if limit is not None and yielded >= limit:
                    break


def _model_tokenizer(model_info: dict) -> str:
    """Read tokenizer metadata, defaulting older in-memory model records to space."""
    return canonical_tokenizer_name(model_info.get("tokenizer", "space"))


class CountNGram:
    """CLI adapter for the shared NgramLM count tables."""

    def __init__(
        self,
        order: int,
        vocab: list[str],
        sentences: Iterable[list[str]],
        smoothing: str = "mle",
        k: float = 1.0,
        discount: float = 0.75,
        allowed_target_ids=None,
    ):
        self.order = order
        self.smoothing = smoothing
        self.k = k
        self.discount = discount
        self.stoi = {word: index for index, word in enumerate(vocab)}
        self.itos = vocab
        self.valid_target_ids = [
            i for i, word in enumerate(vocab) if word not in {PAD, BOS}
        ]
        self._backoff_prediction_cache: dict[tuple[int, ...], tuple[float, int | None]] = {}
        self.sentence_count = 0
        self.token_count = 0
        unk_id = self.stoi[UNK]

        def encoded_sentences():
            for sentence_idx, sentence in enumerate(sentences, start=1):
                if sentence_idx % 100_000 == 0:
                    print(f"  encoded training sentences: {sentence_idx:,}", flush=True)
                self.sentence_count += 1
                self.token_count += len(sentence)
                yield [self.stoi.get(word, unk_id) for word in sentence]

        self.model = NgramLM(
            order=order,
            vocab_size=len(vocab),
            bos=self.stoi[BOS],
            eos=self.stoi[EOS],
            smoothing=smoothing,
            k=k,
            discount=discount,
            allowed_target_ids=allowed_target_ids,
        ).fit(encoded_sentences())
        self.counts = self.model.ngram_counts[order]
        self._best_followers: dict[tuple[int, ...], tuple[int, int]] = {}
        for gram, count in self.counts.items():
            context, token = gram[:-1], gram[-1]
            current = self._best_followers.get(context)
            if current is None or count > current[0] or (count == current[0] and token < current[1]):
                self._best_followers[context] = (count, token)
        self._unigram_rank = sorted(
            self.valid_target_ids,
            key=lambda token: (self.model.ngram_counts[1][(token,)], -token),
            reverse=True,
        )

    def _best_backoff(self, context: tuple[int, ...], excluded=frozenset()):
        """Return the exact highest-probability continuation under recursive backoff."""
        n = len(context) + 1
        if n == 1:
            token = next((word for word in self._unigram_rank if word not in excluded), None)
            return (self.model.prob((token,)), token) if token is not None else (0.0, None)

        followers = self.model.cont_following[n].get(context, ())
        candidates = [
            (self.model.prob(context + (word,)), word)
            for word in followers
            if word not in excluded
        ]
        lower_excluded = excluded.union(followers)
        lower_prob, lower_word = self._best_backoff(context[1:], lower_excluded)
        if lower_word is not None:
            if self.model.context_totals[n][context]:
                lower_prob *= self.model._alpha(n, context)
            candidates.append((lower_prob, lower_word))
        if not candidates:
            return 0.0, None
        return max(candidates, key=lambda pair: (pair[0], -pair[1]))

    def score(self, context: tuple[int, ...], target: int) -> tuple[float, int | None]:
        probability = self.model.prob(context + (target,))
        if self.smoothing == "backoff":
            if context not in self._backoff_prediction_cache:
                self._backoff_prediction_cache[context] = self._best_backoff(context)
            _, predicted = self._backoff_prediction_cache[context]
        else:
            best = self._best_followers.get(context)
            if best is not None:
                predicted = best[1]
            elif self.smoothing == "laplace":
                predicted = min(self.valid_target_ids)
            else:
                predicted = None
        return probability, predicted

    def prediction_probability(self, context: tuple[int, ...]) -> float:
        """Return the probability assigned to the model's top continuation."""
        if self.smoothing == "backoff":
            if context not in self._backoff_prediction_cache:
                self._backoff_prediction_cache[context] = self._best_backoff(context)
            return self._backoff_prediction_cache[context][0]
        best = self._best_followers.get(context)
        if best is None:
            return 1.0 / self.model.V if self.smoothing == "laplace" else 0.0
        count, _ = best
        if self.smoothing == "laplace":
            return (count + self.k) / (
                self.model.context_totals[self.order][context] + self.k * self.model.V
            )
        return count / self.model.context_totals[self.order][context]


class LanguageModelCLI:
    """Interactive shell; models are loaded independently and cached for the session."""

    def __init__(self, device_name: str = "auto"):
        self.device = get_device(device_name)
        self.neural: dict[str, dict] = {}
        self.ngrams: dict[str, dict] = {}
        self.active_neural: str | None = None

    @staticmethod
    def _print_table(columns: list[str], rows: list[list[str]]) -> None:
        """Print rows as a width-aligned plain-text table."""
        widths = [
            max([len(column), *(len(str(row[index])) for row in rows)])
            for index, column in enumerate(columns)
        ]
        border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

        def print_row(values):
            cells = " | ".join(
                f"{str(value):<{width}}" for value, width in zip(values, widths)
            )
            print(f"| {cells} |")

        print(border)
        print_row(columns)
        print(border)
        for row in rows:
            print_row(row)
        print(border)

    @torch.no_grad()
    def compare_csv(self, path: Path, ngram_info: dict, neural_info: dict, limit: int | None = None) -> None:
        """Score rare n-gram CSV entries with a matching n-gram and neural model."""
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        if ngram_info["vocab"] != neural_info["vocab"]:
            raise ValueError("Vocabulary differs; rebuild the n-gram using the neural model's training vocabulary")
        if ngram_info["lowercase"] != neural_info["lowercase"]:
            raise ValueError("The models use different lowercase preprocessing")
        if _model_tokenizer(ngram_info) != _model_tokenizer(neural_info):
            raise ValueError("The models use different tokenizers")

        expected_order = ngram_info["order"]
        rows = []
        skipped_order = skipped_invalid = skipped_eos = 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"sequence", "order"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("Rare n-gram CSV must include 'order' and 'sequence' columns")

            for row in reader:
                try:
                    row_order = int(row["order"])
                    tokens = row["sequence"].split()
                except (AttributeError, TypeError, ValueError):
                    skipped_invalid += 1
                    continue
                if row_order != expected_order:
                    skipped_order += 1
                    continue
                csv_tokenizer = canonical_tokenizer_name(row.get("tokenizer") or "space")
                if csv_tokenizer != _model_tokenizer(ngram_info):
                    skipped_invalid += 1
                    continue
                if len(tokens) != expected_order:
                    skipped_invalid += 1
                    continue
                if tokens[-1] == EOS:
                    skipped_eos += 1
                    continue
                if limit is not None and len(rows) >= limit:
                    continue

                if ngram_info["lowercase"]:
                    tokens = [token.lower() for token in tokens]
                context_words, expected_word = tokens[:-1], tokens[-1]
                ngram_stoi = ngram_info["stoi"]
                ngram_unk = ngram_stoi[UNK]
                context_ids = [ngram_stoi.get(word, ngram_unk) for word in context_words]
                context_ids = [ngram_stoi[BOS]] * (expected_order - 1 - len(context_ids)) + context_ids
                expected_id = ngram_stoi.get(expected_word, ngram_unk)
                context = tuple(context_ids)
                ngram_probability, ngram_prediction = ngram_info["lm"].score(context, expected_id)
                ngram_top_probability = ngram_info["lm"].prediction_probability(context)

                neural_stoi = neural_info["stoi"]
                neural_unk = neural_stoi[UNK]
                neural_context_words = list(context_words)
                while (
                    len(neural_context_words) > 1
                    and neural_context_words[0] == BOS
                    and neural_context_words[1] == BOS
                ):
                    neural_context_words.pop(0)
                neural_context = [neural_stoi.get(word, neural_unk) for word in neural_context_words]
                if not neural_context_words or neural_context_words[0] != BOS:
                    neural_context.insert(0, neural_stoi[BOS])
                neural_input = torch.tensor(
                    [neural_context],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = mask_logits(
                    neural_info["model"](neural_input)[:, -1, :].float(),
                    neural_info.get("allowed_target_ids"),
                )
                probabilities = torch.softmax(logits, dim=-1)[0]
                neural_prediction = int(probabilities.argmax().item())
                neural_expected_probability = float(probabilities[neural_stoi.get(expected_word, neural_unk)].item())

                rows.append([
                    str(expected_order),
                    " ".join(context_words),
                    expected_word,
                    ngram_info["vocab"][ngram_prediction] if ngram_prediction is not None else "<unseen>",
                    f"{ngram_probability:.6g}",
                    f"{ngram_top_probability:.6g}",
                    neural_info["vocab"][neural_prediction],
                    f"{neural_expected_probability:.6g}",
                    row.get("train_count", ""),
                    row.get("test_count", ""),
                ])

        print(f"Rare n-gram CSV: {self._display_path(path)}")
        print(f"Rows scored: {len(rows):,}; skipped for different order: {skipped_order:,}; invalid rows: {skipped_invalid:,}")
        print(f"Sentence-ending targets skipped: {skipped_eos:,}")
        if not rows:
            print(f"No valid {expected_order}-gram rows found in the CSV.")
            return
        self._print_table(
            [
                "Order", "Context", "Expected", "N-gram prediction", "N-gram P(expected)",
                "N-gram P(top)", "Neural prediction", "Neural P(expected)", "Train count", "Test count",
            ],
            rows,
        )

    def compare_ngram_models(
        self,
        path: Path,
        first: dict,
        second: dict,
        limit: int = 25,
        progress_seconds: float = 10.0,
    ) -> None:
        """Evaluate two n-gram models on the same file and show paired results."""
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation file not found: {path}")
        if first["vocab"] != second["vocab"]:
            raise ValueError("Vocabulary differs; build both n-gram models from the same training vocabulary")
        if first["lowercase"] != second["lowercase"]:
            raise ValueError("The models use different lowercase preprocessing")
        if _model_tokenizer(first) != _model_tokenizer(second):
            raise ValueError("The models use different tokenizers")
        if limit < 0 or progress_seconds < 0:
            raise ValueError("--limit and --print-progress must be zero or positive")

        vocab = Vocab.from_list(first["vocab"])
        samples = [{}, {}]
        started = time.perf_counter()
        last_progress = [started, started]

        def make_sink(slot, model_info):
            count = 0

            def retain(event):
                nonlocal count
                count += 1
                now = time.perf_counter()
                if progress_seconds and now - last_progress[slot] >= progress_seconds:
                    print(
                        f"  {model_info['name']} progress: {count:,} events "
                        f"({now - started:.1f}s elapsed)",
                        flush=True,
                    )
                    last_progress[slot] = now
                if event["eos"] or (limit and len(samples[slot]) >= limit):
                    return
                key = (event["sentence_index"], event["position"])
                probability = math.exp(event["log_probability"]) if math.isfinite(event["log_probability"]) else 0.0
                samples[slot][key] = {
                    "expected": first["vocab"][event["target"]],
                    "prediction": first["vocab"][event["prediction"]] if event["prediction"] is not None else "<unseen>",
                    "probability": probability,
                    "ids": event["ids"],
                    "position": event["position"],
                }

            return retain

        results = []
        for slot, model_info in enumerate((first, second)):
            results.append(evaluate_model(
                model_info["lm"], vocab,
                iter_sentences(path, model_info["lowercase"], tokenizer_name=_model_tokenizer(model_info)), "ngram",
                event_sink=make_sink(slot, model_info), case_limit=0,
            ))

        total = results[0]["tokens"]
        words = results[0]["word_tokens"]
        oov = results[0]["oov_tokens"]
        print(f"Evaluation finished in {time.perf_counter() - started:.1f}s.")
        print(f"\nCompared on {self._display_path(path)}")
        print(
            f"Sentences: {results[0]['sentences']:,}; word tokens: {words:,}; "
            f"perplexity events (including </s>): {total:,}; OOV: {oov:,}/{words:,} "
            f"({oov / words:.2%})" if words else
            f"Sentences: {results[0]['sentences']:,}; perplexity events: {total:,}; OOV: 0"
        )
        print(
            f"Sentence-ending targets excluded from word prediction metrics: "
            f"{results[0]['sentence_end_targets']:,}"
        )

        metric_rows = []
        for label, result in zip((first["name"], second["name"]), results):
            word_result = result["word_metrics"]
            accuracy = word_result["accuracy"] or 0.0
            if result["perplexity"] is not None:
                perplexity = f"{result['perplexity']:.5g}"
            elif result["zero_probability_events"]:
                perplexity = f"infinite ({result['zero_probability_events']:,} zero events)"
            else:
                perplexity = result["status"]
            coverage = (
                word_result["context_covered"] / word_result["tokens"]
                if word_result["tokens"] else 0.0
            )
            metric_rows.append([
                label,
                f"{word_result['correct']:,}/{word_result['tokens']:,} ({accuracy:.2%})",
                f"{coverage:.2%}",
                f"{result['zero_probability_events']:,}/{result['tokens']:,}",
                perplexity,
            ])
        self._print_table(
            ["Model", "Word correct / top-1 accuracy", "Word context coverage", "Zero-probability (incl. </s>)", "Perplexity (incl. </s>)"],
            metric_rows,
        )

        keys = sorted(set(samples[0]) | set(samples[1]))
        if limit:
            keys = keys[:limit]
        width = max(first["order"], second["order"]) - 1
        paired_rows = []
        for key in keys:
            record = samples[0].get(key) or samples[1].get(key)
            prefix = [vocab.stoi[BOS]] * width + record["ids"][:record["position"]]
            context = " ".join(first["vocab"][token] for token in prefix[-width:]) if width else ""
            one, two = samples[0].get(key, {}), samples[1].get(key, {})
            paired_rows.append([
                context,
                record["expected"],
                one.get("prediction", "—"),
                f"{one['probability']:.6g}" if "probability" in one else "—",
                two.get("prediction", "—"),
                f"{two['probability']:.6g}" if "probability" in two else "—",
            ])
        print(f"\nPaired word predictions (first {len(paired_rows):,}; sentence-ending targets omitted)")
        if paired_rows:
            self._print_table(
                ["Context", "Expected", first["name"] + " prediction", first["name"] + " P(expected)",
                 second["name"] + " prediction", second["name"] + " P(expected)"],
                paired_rows,
            )
        else:
            print("(no word targets)")

    def compare_ngram_csv(self, path: Path, first: dict, second: dict, limit: int | None = None) -> None:
        """Score rare CSV rows against two same-order n-gram models."""
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        if first["order"] != second["order"]:
            raise ValueError("Rare CSV comparison requires both n-gram models to have the same order")
        if first["vocab"] != second["vocab"]:
            raise ValueError("Vocabulary differs; build both n-gram models from the same training vocabulary")
        if first["lowercase"] != second["lowercase"]:
            raise ValueError("The models use different lowercase preprocessing")
        if _model_tokenizer(first) != _model_tokenizer(second):
            raise ValueError("The models use different tokenizers")

        rows = []
        skipped_order = skipped_invalid = skipped_eos = 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"order", "sequence"}.issubset(reader.fieldnames):
                raise ValueError("Rare n-gram CSV must include 'order' and 'sequence' columns")
            for row in reader:
                try:
                    row_order = int(row["order"])
                    tokens = row["sequence"].split()
                except (AttributeError, TypeError, ValueError):
                    skipped_invalid += 1
                    continue
                if row_order != first["order"]:
                    skipped_order += 1
                    continue
                if canonical_tokenizer_name(row.get("tokenizer") or "space") != _model_tokenizer(first):
                    skipped_invalid += 1
                    continue
                if len(tokens) != row_order:
                    skipped_invalid += 1
                    continue
                if tokens[-1] == EOS:
                    skipped_eos += 1
                    continue
                if limit is not None and len(rows) >= limit:
                    continue
                if first["lowercase"]:
                    tokens = [token.lower() for token in tokens]
                context_words, expected_word = tokens[:-1], tokens[-1]
                ids = first["stoi"]
                context = tuple(ids.get(word, ids[UNK]) for word in context_words)
                target = ids.get(expected_word, ids[UNK])
                current = []
                for model in (first, second):
                    probability, prediction = model["lm"].score(context, target)
                    current.extend([
                        model["vocab"][prediction] if prediction is not None else "<unseen>",
                        f"{probability:.6g}",
                    ])
                rows.append([
                    " ".join(context_words), expected_word,
                    *current,
                    row.get("train_count", ""), row.get("test_count", ""),
                ])
        print(f"Rare n-gram CSV: {self._display_path(path)}")
        print(f"Rows scored: {len(rows):,}; skipped for different order: {skipped_order:,}; invalid rows: {skipped_invalid:,}")
        print(f"Sentence-ending targets skipped: {skipped_eos:,}")
        if rows:
            self._print_table(
                ["Context", "Expected", first["name"] + " prediction", first["name"] + " P(expected)",
                 second["name"] + " prediction", second["name"] + " P(expected)", "Train count", "Test count"],
                rows,
            )
        else:
            print(f"No valid {first['order']}-gram rows found in the CSV.")

    @staticmethod
    def _display_path(path: Path) -> str:
        """Prefer repository-relative paths in CLI output."""
        project_root = Path(__file__).resolve().parents[2]
        try:
            return path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _read_run(path: Path):
        config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        if not isinstance(vocab, list) or not {PAD, UNK, BOS, EOS}.issubset(vocab):
            raise ValueError(f"Invalid vocab.json in {path}")
        return config, vocab

    def load_neural(self, path: Path):
        path = path.expanduser().resolve()
        if path.is_dir():
            candidates = sorted(path.glob("*_seed*.pt"))
            if len(candidates) != 1:
                raise ValueError(f"Pass one checkpoint file; found {len(candidates)} in {path}")
            path = candidates[0]
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        config, vocab = self._read_run(path.parent)
        stoi = {w: i for i, w in enumerate(vocab)}
        name = path.stem.rsplit("_seed", 1)[0]
        context_match = re.fullmatch(r"ffnn_ctx(\d+)", name)
        model_type = "ffnn" if context_match else name
        model_config = next(
            (m for m in config.get("models", []) if m["name"] == model_type), None
        )
        if model_config is None:
            raise ValueError(f"No matching model configuration for {path.name}")
        model_params = dict(model_config.get("params", {}))
        if context_match:
            model_params["context_size"] = int(context_match.group(1))
        model = build_model(model_type, len(vocab), stoi[PAD], **model_params)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.to(self.device).eval()
        data = config.get("data", {})
        source_train_path = Path(data.get("output", "data/processed")) / data.get("corpus", "") / "train.txt"
        if not source_train_path.is_absolute():
            source_train_path = Path(__file__).resolve().parents[2] / source_train_path
        # Percentage-specific runs keep the exact lines consumed by training
        # beside their artifacts, so downstream count models can reuse them.
        snapshot_path = next(
            (parent / "train.txt" for parent in path.parents if (parent / "train.txt").is_file()),
            None,
        )
        train_path = snapshot_path or source_train_path
        train_percent = float(config.get("train_percent", 100))
        info = {
            "name": name, "model": model, "checkpoint": path, "vocab": vocab, "stoi": stoi,
            "run_dir": path.parent,
            "lowercase": data.get("lowercase", True),
            "tokenizer": canonical_tokenizer_name(data.get("tokenizer", "space")),
            "train_path": train_path.resolve(),
            "train_percent": train_percent,
            "train_file_percent": 100.0 if snapshot_path else train_percent,
        }
        if "allowed_target_ids" in config:
            info["allowed_target_ids"] = config["allowed_target_ids"]
        key = str(path)
        self.neural[key] = info
        self.active_neural = key
        print(
            f"Loaded neural {name}: {path.name} (vocab {len(vocab):,}, "
            f"tokenizer={info['tokenizer']}, device {self.device})"
        )
        return key

    def _active_model(self):
        if self.active_neural is None:
            raise ValueError("Load a neural checkpoint first with: nn load CHECKPOINT.pt")
        return self.neural[self.active_neural]

    def show_vocab(self, limit: int = 10):
        info = self._active_model()
        words = [w for w in info["vocab"] if w not in {PAD, UNK, BOS, EOS}]
        print(f"Top {min(limit, len(words))} vocabulary entries (frequency order):")
        for rank, word in enumerate(words[:limit], start=1):
            print(f"  {rank:>2}. {word}")
        print(f"Vocabulary size: {len(info['vocab']):,} including special tokens")

    def show_stats(self):
        info = self._active_model()
        path = info["run_dir"] / "data_stats.json"
        stats = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        print(f"Model: {info['name']} ({info['checkpoint'].name})")
        print(f"Vocabulary size: {len(info['vocab']):,}")
        print(f"Training split: {info['train_path']}")
        for key, value in stats.items():
            print(f"{key}: {value:.2%}" if "oov_rate" in key and isinstance(value, (int, float)) else f"{key}: {value}")

    @torch.no_grad()
    def predict(self, text: str, limit: int = 5):
        info = self._active_model()
        words = tokenize(text, _model_tokenizer(info), lowercase=info["lowercase"])
        stoi, vocab = info["stoi"], info["vocab"]
        oov = sum(w not in stoi for w in words)
        ids = [stoi[BOS]] + [stoi.get(w, stoi[UNK]) for w in words]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = info["model"](x)[0, -1].float()
        logits[stoi[PAD]] = -torch.inf
        logits[stoi[BOS]] = -torch.inf
        probs = torch.softmax(logits, dim=-1)
        values, indexes = torch.topk(probs, min(limit, len(vocab)))
        if oov:
            print(f"Prompt OOV words mapped to <unk>: {oov}/{len(words)}")
        for prob, index in zip(values.tolist(), indexes.tolist()):
            print(f"{vocab[index]:<20} {prob:.4f}")

    @staticmethod
    def _read_sentences(path: Path):
        """Compatibility wrapper for reading one prepared sentence per line."""
        yield from iter_sentences(path, lowercase=False)

    def load_ngram(
        self,
        name: str,
        order: int,
        train_path: Path,
        min_freq: int = 2,
        smoothing: str = "mle",
        k: float | None = None,
        discount: float = 0.75,
        tokenizer_name: str | None = None,
    ):
        if order < 2:
            raise ValueError("n must be at least 2")
        if smoothing == "laplace" and k is None:
            raise ValueError("Laplace smoothing requires --k, for example --smoothing laplace --k 1.0")
        if smoothing == "laplace" and k <= 0:
            raise ValueError("Laplace --k must be greater than zero")
        if smoothing == "backoff" and not 0 <= discount < 1:
            raise ValueError("Backoff --discount must be at least 0 and less than 1")
        if smoothing not in {"mle", "laplace", "backoff"}:
            raise ValueError("--smoothing must be mle, laplace, or backoff")
        train_path = train_path.expanduser().resolve()
        if not train_path.is_file():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        # Match the saved neural vocabulary exactly whenever the same train split is used.
        nn_info = next((m for m in reversed(list(self.neural.values())) if m["train_path"] == train_path), None)
        nn_tokenizer = _model_tokenizer(nn_info) if nn_info else "space"
        tokenizer_name = canonical_tokenizer_name(
            tokenizer_name if tokenizer_name is not None
            else nn_tokenizer
        )
        lowercase = nn_info["lowercase"] if nn_info else True
        train_percent = (
            nn_info.get("train_file_percent", nn_info["train_percent"])
            if nn_info else 100
        )
        use_neural_vocab = nn_info is not None and nn_tokenizer == tokenizer_name
        if use_neural_vocab:
            vocab = nn_info["vocab"]
        else:
            counts = Counter()
            total = 0
            for words in iter_sentences(train_path, lowercase, train_percent, tokenizer_name):
                counts.update(words)
                total += 1
                if total % 100_000 == 0:
                    print(f"  vocabulary scan: {total:,} sentences", flush=True)
            vocab = Vocab.from_counts(counts, min_freq=min_freq).itos
        smoothing_label = (
            f"Laplace k={k:g}" if smoothing == "laplace"
            else f"backoff discount={discount:g}" if smoothing == "backoff"
            else "unsmoothed MLE"
        )
        print(
            f"Building {smoothing_label} {order}-gram '{name}' from "
            f"{self._display_path(train_path)} (tokenizer={tokenizer_name})",
            flush=True,
        )
        started_at = time.perf_counter()
        allowed_target_ids = None
        if use_neural_vocab:
            allowed_target_ids = nn_info.get("allowed_target_ids")
        lm = CountNGram(
            order,
            vocab,
            iter_sentences(train_path, lowercase, train_percent, tokenizer_name),
            smoothing=smoothing,
            k=1.0 if k is None else k,
            discount=discount,
            allowed_target_ids=allowed_target_ids,
        )
        self.ngrams[name] = {
            "name": name, "order": order, "lm": lm, "vocab": vocab,
            "stoi": {w: i for i, w in enumerate(vocab)}, "lowercase": lowercase,
            "tokenizer": tokenizer_name,
            "train_path": train_path,
            "smoothing": smoothing, "k": k, "discount": discount,
        }
        stats = [
            ("Status", "ready in this session"),
            ("Name", name),
            ("Order", str(order)),
            ("Smoothing", smoothing_label),
            ("Tokenizer", tokenizer_name),
            ("Training file", self._display_path(train_path)),
            ("Sentences", f"{lm.sentence_count:,}"),
            ("Training tokens", f"{lm.token_count:,}"),
            ("Vocabulary size", f"{len(vocab):,}"),
            (f"Distinct {order}-grams", f"{len(lm.counts):,}"),
            ("Counting time", f"{time.perf_counter() - started_at:.1f}s"),
            ("N-grams in session", ", ".join(self.ngrams)),
        ]
        metric_width = max(len(metric) for metric, _ in stats)
        print("\nN-gram build complete")
        print(f"{'Metric':<{metric_width}} | Value")
        print(f"{'-' * metric_width}-+------------------------------")
        for metric, value in stats:
            print(f"{metric:<{metric_width}} | {value}")

    @staticmethod
    def _sentence_events(ids: list[int], order: int, bos: int, eos: int):
        yield from sentence_events(ids, order, bos, eos)

    @torch.no_grad()
    def _neural_scores(self, model_info: dict, contexts: list[list[int]], targets: list[int]):
        x = torch.tensor(contexts, dtype=torch.long, device=self.device)
        logits = mask_logits(
            model_info["model"](x)[:, -1, :].float(), model_info.get("allowed_target_ids")
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        y = torch.tensor(targets, dtype=torch.long, device=self.device)
        losses = -log_probs.gather(1, y[:, None]).squeeze(1)
        return losses.cpu().tolist(), logits.argmax(-1).cpu().tolist()

    def evaluate(
        self,
        path: Path,
        ngram_info: dict | None,
        neural_info: dict | None,
        limit: int = 25,
        progress_seconds: float = 10.0,
    ):
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation file not found: {path}")
        if ngram_info is None and neural_info is None:
            raise ValueError("No model selected")
        base = ngram_info or neural_info
        vocab, stoi = base["vocab"], base["stoi"]
        lowercase = base["lowercase"]
        if ngram_info and neural_info and ngram_info["vocab"] != neural_info["vocab"]:
            raise ValueError("Vocabulary differs. Load the neural checkpoint first, then rebuild the n-gram on its train.txt")
        if ngram_info and neural_info and ngram_info["lowercase"] != neural_info["lowercase"]:
            raise ValueError("The models use different lowercase preprocessing")
        if ngram_info and neural_info and _model_tokenizer(ngram_info) != _model_tokenizer(neural_info):
            raise ValueError("The models use different tokenizers")
        if progress_seconds < 0:
            raise ValueError("--print-progress must be zero or a positive number of seconds")
        if limit < 0:
            raise ValueError("--limit must be zero or a positive number")
        started_at = time.perf_counter()
        last_progress_at = started_at
        next_event_check = 10_000
        if ngram_info:
            smoothing = ngram_info["smoothing"]
            smoothing_desc = (
                f"Laplace k={ngram_info['k']:g}" if smoothing == "laplace"
                else f"backoff d={ngram_info['discount']:g}" if smoothing == "backoff"
                else "MLE"
            )
            model_label = f"{ngram_info['name']} ({ngram_info['order']}-gram {smoothing_desc})"
        else:
            smoothing_desc = ""
            model_label = neural_info["name"]
        model_label += f" and {neural_info['name']}" if ngram_info and neural_info else ""
        progress_note = (
            f"progress updates about every {progress_seconds:g}s."
            if progress_seconds else "periodic progress disabled."
        )
        print(f"Evaluating {path.name} with {model_label}; {progress_note}", flush=True)
        bos, eos, unk = stoi[BOS], stoi[EOS], stoi[UNK]
        order = ngram_info["order"] if ngram_info else None
        neural_rows = {}
        counters = {"ngram": 0, "neural": 0}
        metric_vocab = Vocab.from_list(vocab)
        correct_samples: dict[int, dict] = {}
        incorrect_samples: dict[int, dict] = {}
        correct_heap: list[tuple[float, int, int, dict]] = []
        incorrect_heap: list[tuple[float, int, int, dict]] = []
        selected_by_event: dict[tuple[int, int], dict] = {}
        # Track only retained contexts, keeping this bounded by 2 * limit.
        # With --limit 0, all unique contexts are intentionally retained.
        used_contexts = {"correct": {}, "incorrect": {}}

        def retain(event, family):
            nonlocal last_progress_at
            row_index = counters[family]
            counters[family] += 1
            now = time.perf_counter()
            if row_index % 10_000 == 0 and progress_seconds and now - last_progress_at >= progress_seconds:
                print(f"  evaluation progress ({family}): {row_index:,} events ({now - started_at:.1f}s elapsed)", flush=True)
                last_progress_at = now
            # Keep EOS in aggregate accuracy/perplexity, but omit these obvious
            # sentence-boundary examples from the prediction tables.
            if event["target"] == eos:
                return
            prediction = event["prediction"]
            log_probability = event["log_probability"]
            probability = math.exp(log_probability) if math.isfinite(log_probability) else 0.0
            event_key = (event["sentence_index"], event["position"])
            if family == "neural":
                if ngram_info:
                    record = selected_by_event.get(event_key)
                    if record is not None:
                        record["neural_prediction"] = prediction
                        record["neural_expected_probability"] = probability
                elif limit == 0 or row_index < limit:
                    width = 1
                    prefix = [bos] + event["ids"][:event["position"]]
                    neural_rows[row_index] = [
                        " ".join(vocab[token] for token in prefix[-width:]),
                        vocab[event["target"]],
                        vocab[prediction] if prediction is not None else "<unseen>",
                        f"{probability:.6g}",
                    ]
                return

            width = order - 1
            prefix = [bos] * width + event["ids"][:event["position"]]
            context = tuple(prefix[-width:])
            correct = prediction == event["target"]
            sample_group = "correct" if correct else "incorrect"
            if context in used_contexts[sample_group]:
                return
            record = {
                "index": row_index,
                "key": event_key,
                "context_key": context,
                "context": " ".join(vocab[token] for token in context),
                "expected": vocab[event["target"]],
                "prediction_text": vocab[prediction] if prediction is not None else "<unseen>",
                "expected_probability": probability,
                "prediction_probability": ngram_info["lm"].prediction_probability(context),
                "correct": correct,
            }
            samples = correct_samples if record["correct"] else incorrect_samples
            heap = correct_heap if record["correct"] else incorrect_heap
            if limit == 0:
                samples[row_index] = record
                used_contexts[sample_group][context] = row_index
                selected_by_event[event_key] = record
            elif len(heap) < limit:
                heapq.heappush(heap, (record["prediction_probability"], -row_index, row_index, record))
                samples[row_index] = record
                used_contexts[sample_group][context] = row_index
                selected_by_event[event_key] = record
            elif (record["prediction_probability"], -row_index) > heap[0][:2]:
                removed = heapq.heapreplace(
                    heap,
                    (record["prediction_probability"], -row_index, row_index, record),
                )
                samples.pop(removed[2], None)
                used_contexts[sample_group].pop(removed[3]["context_key"], None)
                selected_by_event.pop(removed[3]["key"], None)
                samples[row_index] = record
                used_contexts[sample_group][context] = row_index
                selected_by_event[event_key] = record

        ng_result = nn_result = None
        ng_nll = nn_nll = 0.0
        ng_zero = ng_correct = ng_covered = nn_correct = 0
        if ngram_info:
            ng_result = evaluate_model(
                ngram_info["lm"], metric_vocab,
                iter_sentences(path, lowercase, tokenizer_name=_model_tokenizer(ngram_info)), "ngram",
                event_sink=lambda event: retain(event, "ngram"), case_limit=0,
            )
            ng_nll = ng_result["finite_nll_sum"]
            ng_zero = ng_result["zero_probability_events"]
            ng_correct = ng_result["correct"]
            ng_covered = ng_result["context_covered"]
        if neural_info:
            nn_result = evaluate_model(
                neural_info["model"], metric_vocab,
                iter_sentences(path, lowercase, tokenizer_name=_model_tokenizer(neural_info)), "neural", self.device,
                allowed_target_ids=neural_info.get("allowed_target_ids"),
                event_sink=lambda event: retain(event, "neural"), case_limit=0,
            )
            nn_nll = nn_result["finite_nll_sum"]
            nn_correct = nn_result["correct"]
        base_result = ng_result or nn_result
        total = base_result["tokens"]
        sentence_count = base_result["sentences"]
        words_count = base_result["word_tokens"]
        oov = base_result["oov_tokens"]
        print(f"Evaluation finished in {time.perf_counter() - started_at:.1f}s.", flush=True)
        print("\nEvaluation summary")
        print(f"  File: {self._display_path(path)}")
        print(f"  Sentences: {sentence_count:,}")
        print(f"  Word tokens: {words_count:,}")
        print(
            f"  Perplexity events: {total:,} "
            f"({words_count:,} word targets + {sentence_count:,} sentence-ending targets)"
        )
        print(f"  Sentence-ending targets excluded from word prediction metrics: {sentence_count:,}")
        print(
            f"  OOV word tokens: {oov:,}/{words_count:,} ({oov / words_count:.2%})"
            if words_count else "  OOV word tokens: 0"
        )
        print("\nModel results")
        model_results = []

        def perplexity_text(result):
            if result["perplexity"] is not None:
                return f"{result['perplexity']:.4g}"
            if result["zero_probability_events"]:
                return f"infinite ({result['zero_probability_events']:,} zero-probability events)"
            return result["status"]

        if ngram_info:
            ng_word_result = ng_result["word_metrics"]
            accuracy = ng_word_result["accuracy"]
            seen_accuracy = math.nan
            if ng_word_result["context_covered"]:
                seen_accuracy = (
                    ng_word_result["correct_when_context_seen"]
                    / ng_word_result["context_covered"]
                )
            context_coverage = (
                f"{ng_word_result['context_covered']:,} / {words_count:,} "
                f"({ng_word_result['context_covered'] / words_count:.4%})" if words_count else "n/a"
            )
            word_correct_text = (
                f"{ng_word_result['correct']:,} / {words_count:,} ({accuracy:.2%})"
                if accuracy is not None else "n/a (no word targets)"
            )
            model_results.append((
                f"{ngram_info['name']} ({order}-gram, {smoothing_desc})",
                [
                    ("Correct word predictions", word_correct_text),
                    ("Top-1 accuracy (word targets)", f"{accuracy:.2%}" if accuracy is not None else "n/a"),
                    ("Top-1 when context seen", f"{seen_accuracy:.2%}" if ng_word_result["context_covered"] else "n/a"),
                    ("Contexts seen in training", context_coverage),
                    ("Zero-probability events (incl. </s>)", f"{ng_zero:,} / {total:,} ({ng_zero / total:.2%})" if total else "n/a"),
                    ("Perplexity (incl. </s>)", perplexity_text(ng_result)),
                ] if total else [("Evaluation", "No events")],
            ))
        if neural_info:
            nn_word_result = nn_result["word_metrics"]
            accuracy = nn_word_result["accuracy"]
            nn_zero = nn_result["zero_probability_events"]
            word_correct_text = (
                f"{nn_word_result['correct']:,} / {words_count:,} ({accuracy:.2%})"
                if accuracy is not None else "n/a (no word targets)"
            )
            model_results.append((
                f"{neural_info['name']} ({neural_info['checkpoint'].name})",
                [
                    ("Correct word predictions", word_correct_text),
                    ("Top-1 accuracy (word targets)", f"{accuracy:.2%}" if accuracy is not None else "n/a"),
                    ("Top-1 when context seen", "n/a (neural model)"),
                    ("Contexts seen in training", "n/a (neural model)"),
                    ("Zero-probability events (incl. </s>)", f"{nn_zero:,} / {total:,} ({nn_zero / total:.2%})"),
                    ("Perplexity (incl. </s>)", perplexity_text(nn_result)),
                ] if total else [("Evaluation", "No events")],
            ))

        for model_name, metrics in model_results:
            print(f"\n{model_name}")
            metric_width = max(len("Metric"), *(len(label) for label, _ in metrics))
            value_width = max(len("Result"), *(len(value) for _, value in metrics))
            border = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
            print(border)
            print(f"| {'Metric':<{metric_width}} | {'Result':<{value_width}} |")
            print(border)
            for label, value in metrics:
                print(f"| {label:<{metric_width}} | {value:<{value_width}} |")
            print(border)

        print("\nTop-1 accuracy and context coverage use word targets only; sentence-ending targets are excluded.")
        print("Perplexity and zero-probability counts include sentence-ending targets (</s>).")
        if ngram_info and ng_zero and ngram_info["smoothing"] == "mle":
            print("An unsmoothed n-gram assigns zero probability to some targets, making perplexity infinite.")
        if ngram_info:
            columns = ["Context", "Expected", "N-gram prediction", "P(expected)", "P(prediction)"]
            if neural_info:
                columns.extend(("Neural prediction", "Neural P(expected)"))
            for title, samples in (
                ("Correct n-gram predictions", correct_samples),
                ("Incorrect n-gram predictions", incorrect_samples),
            ):
                ranked = sorted(
                    samples.values(),
                    key=lambda row: (-row["prediction_probability"], row["index"]),
                )
                shown = "all unique contexts" if limit == 0 else f"top {limit} unique contexts"
                print(f"\n{title} ({shown}, ranked by P(prediction))")
                if not ranked:
                    print("(no examples)")
                    continue
                table_rows = []
                for row in ranked:
                    values = [
                        row["context"], row["expected"], row["prediction_text"],
                        f"{row['expected_probability']:.6g}",
                        f"{row['prediction_probability']:.6g}",
                    ]
                    if neural_info:
                        neural_prediction = row.get("neural_prediction")
                        neural_expected_probability = row.get("neural_expected_probability", math.nan)
                        values.extend((
                            vocab[neural_prediction] if neural_prediction is not None else "?",
                            f"{neural_expected_probability:.6g}",
                        ))
                    table_rows.append(values)
                self._print_table(columns, table_rows)
        else:
            columns = ["Context", "Expected", "Neural prediction", "P(expected)"]
            table_rows = [neural_rows[index] for index in sorted(neural_rows)]
            if table_rows:
                self._print_table(columns, table_rows)
            else:
                print("(no examples)")
        return {"ngram": ng_result, "neural": nn_result}

    def resolve_neural(self, name: str):
        for key, info in self.neural.items():
            if name in {key, info["checkpoint"].name, info["name"]}:
                return info
        return self.neural[self.load_neural(Path(name))]

    def help(self):
        print("Commands:")
        print("  nn load CHECKPOINT.pt")
        print("  vocab [N] | stats | predict TEXT     inspect the active neural checkpoint")
        print("  ngram NAME --n N --train TRAIN.txt [--token space|wp|tb] [--min-freq N] [--smoothing mle|laplace|backoff]")
        print("       Laplace requires --k; backoff accepts --discount (default 0.75).")
        print("  ngram NAME --evaluate FILE.txt [--token space|wp|tb] [--limit N] [--print-progress SECONDS]")
        print("       Shows up to 25 unique correct and incorrect contexts, ranked by P(prediction); --limit 0 shows all unique contexts.")
        print("  compare --one ngram NAME --two ngram|neural NAME (--test FILE.txt | --csv RARE.csv) [--limit N] [--n N]")
        print("       Supports n-gram vs neural, or two same-order n-gram models (for example MLE vs Laplace).")
        print("       --csv scores matching-order rows; --n caps CSV rows (default: all); --limit applies to --test examples.")
        print("  models | help | quit")
        print("  Use Up/Down for command history and Tab for command, model, option, and path completion.")

    def _setup_line_editing(self) -> None:
        """Enable persistent history and tab completion when readline is available."""
        if readline is None:
            print("Line editing unavailable: this Python build has no readline support.")
            return
        self._history_path = Path.home() / ".bits_nlp_lm_history"
        try:
            readline.read_history_file(str(self._history_path))
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Could not load command history: {exc}")
        readline.set_history_length(1000)
        readline.parse_and_bind("tab: complete")
        readline.set_completer_delims(" \t\n")
        readline.set_completer(self._complete)
        if hasattr(readline, "set_auto_history"):
            readline.set_auto_history(True)
        atexit.register(self._save_line_history)

    def _save_line_history(self) -> None:
        if readline is None or not hasattr(self, "_history_path"):
            return
        try:
            readline.write_history_file(str(self._history_path))
        except OSError as exc:
            print(f"Could not save command history: {exc}")

    @staticmethod
    def _path_completions(prefix: str) -> list[str]:
        expanded = os.path.expanduser(prefix)
        matches = []
        for match in glob.glob(expanded + "*"):
            is_directory = os.path.isdir(match)
            if prefix.startswith("~"):
                home = str(Path.home())
                if match.startswith(home):
                    match = "~" + match[len(home):]
            if is_directory:
                match += os.sep
            matches.append(match.replace(" ", "\\ "))
        return matches

    def _complete(self, text: str, state: int):
        if state == 0:
            line = readline.get_line_buffer()
            before = line[:readline.get_begidx()]
            try:
                tokens = shlex.split(before)
            except ValueError:
                tokens = before.split()

            commands = ["nn", "vocab", "stats", "predict", "ngram", "compare", "models", "help", "quit"]
            options = ["--n", "--train", "--evaluate", "--min-freq", "--smoothing", "--k", "--discount", "--limit", "--print-progress", "--token", "--one", "--two", "--test", "--csv"]
            if not tokens:
                choices = commands
            elif tokens[0] == "nn" and len(tokens) == 1:
                choices = ["load"]
            elif tokens[0] == "nn" and len(tokens) >= 2 and tokens[1] == "load":
                choices = self._path_completions(text)
            elif tokens[0] == "ngram" and tokens[-1] in {"--train", "--evaluate"}:
                choices = self._path_completions(text)
            elif tokens[0] == "ngram" and tokens[-1] == "--token":
                choices = list(tokenizer_names())
            elif tokens[0] == "ngram" and tokens[-1] == "--smoothing":
                choices = ["mle", "laplace", "backoff"]
            elif tokens[0] == "ngram" and text.startswith("-"):
                choices = options[:10]
            elif tokens[0] == "ngram" and len(tokens) >= 2 and not text:
                choices = options[:10]
            elif tokens[0] == "compare" and tokens[-1] in {"--test", "--csv"}:
                choices = self._path_completions(text)
            elif tokens[0] == "compare" and tokens[-1] in {"--one", "--two"}:
                choices = ["ngram", "neural"]
            elif (
                tokens[0] == "compare" and len(tokens) >= 2
                and tokens[-2] in {"--one", "--two"} and tokens[-1] in {"ngram", "neural"}
            ):
                choices = (
                    list(self.ngrams)
                    if tokens[-1] == "ngram"
                    else [Path(key).name for key in self.neural] + self._path_completions(text)
                )
            elif tokens[0] == "compare" and text.startswith("-"):
                choices = options
            elif tokens and tokens[-1] in {"--train", "--test", "--csv", "--evaluate"}:
                choices = self._path_completions(text)
            else:
                choices = []
            self._completion_options = [choice for choice in choices if choice.startswith(text)]
        try:
            return self._completion_options[state]
        except IndexError:
            return None

    def shell(self):
        self._setup_line_editing()
        print(f"Ready on {self.device}; no models loaded.")
        self.help()
        while True:
            try:
                raw = input("lm> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            try:
                args = shlex.split(raw)
                if not args:
                    continue
                cmd = args[0].lower()
                if cmd in {"quit", "exit", "q"}:
                    return
                if cmd in {"help", "?"}:
                    self.help()
                elif cmd == "models":
                    print("N-grams:", ", ".join(f"{k}(n={v['order']})" for k, v in self.ngrams.items()) or "(none)")
                    print("Neural:", ", ".join(Path(k).name for k in self.neural) or "(none)")
                elif cmd == "nn" and len(args) == 3 and args[1] == "load":
                    self.load_neural(Path(args[2]))
                elif cmd == "vocab":
                    self.show_vocab(int(args[1]) if len(args) > 1 else 10)
                elif cmd == "stats":
                    self.show_stats()
                elif cmd == "predict" and len(args) >= 2:
                    self.predict(" ".join(args[1:]))
                elif cmd == "ngram":
                    parser = argparse.ArgumentParser(prog="ngram", add_help=False)
                    parser.add_argument("name")
                    group = parser.add_mutually_exclusive_group(required=True)
                    group.add_argument("--n", type=int)
                    group.add_argument("--evaluate", type=Path)
                    parser.add_argument("--train", type=Path)
                    parser.add_argument("--min-freq", type=int, default=2)
                    parser.add_argument("--smoothing", choices=("mle", "laplace", "backoff"), default="mle")
                    parser.add_argument("--k", type=float, help="required Laplace add-k value")
                    parser.add_argument("--discount", type=float, default=0.75, help="backoff discount")
                    parser.add_argument("--limit", type=int, default=25)
                    parser.add_argument("--print-progress", type=float, default=10.0)
                    parser.add_argument("--token", choices=tokenizer_names(), default=None)
                    a = parser.parse_args(args[1:])
                    if a.n is not None:
                        if a.train is None:
                            raise ValueError("--n requires --train")
                        self.load_ngram(
                            a.name, a.n, a.train, a.min_freq,
                            smoothing=a.smoothing, k=a.k, discount=a.discount,
                            tokenizer_name=a.token,
                        )
                    else:
                        if a.name not in self.ngrams:
                            raise ValueError(f"Unknown n-gram: {a.name}")
                        if a.token and canonical_tokenizer_name(a.token) != _model_tokenizer(self.ngrams[a.name]):
                            raise ValueError("--token must match the tokenizer used to build this n-gram")
                        self.evaluate(a.evaluate, self.ngrams[a.name], None, a.limit, a.print_progress)
                elif cmd == "compare":
                    parser = argparse.ArgumentParser(prog="compare", add_help=False)
                    parser.add_argument("--one", nargs=2, required=True, metavar=("TYPE", "NAME"))
                    parser.add_argument("--two", nargs=2, required=True, metavar=("TYPE", "NAME"))
                    input_group = parser.add_mutually_exclusive_group(required=True)
                    input_group.add_argument("--test", type=Path)
                    input_group.add_argument("--csv", type=Path, help="rare n-gram CSV from analyze_rare_ngrams.py")
                    parser.add_argument("--limit", type=int, default=25)
                    parser.add_argument("--n", type=int, help="maximum matching CSV entries to score; default is all")
                    parser.add_argument("--print-progress", type=float, default=10.0)
                    a = parser.parse_args(args[1:])
                    if a.n is not None and a.n < 1:
                        raise ValueError("--n must be a positive number of CSV entries")
                    if a.n is not None and a.csv is None:
                        raise ValueError("--n applies only when comparing a --csv file")
                    models = []
                    for kind, name in (a.one, a.two):
                        if kind.lower() == "ngram":
                            if name not in self.ngrams:
                                raise ValueError(f"N-gram '{name}' is not loaded")
                            models.append(self.ngrams[name])
                        elif kind.lower() in {"neural", "nn"}:
                            models.append(self.resolve_neural(name))
                        else:
                            raise ValueError("model types are ngram or neural")
                    if len(models) != 2:
                        raise ValueError("compare requires exactly two models")
                    first, second = models
                    first_is_ngram = "order" in first
                    second_is_ngram = "order" in second
                    first_is_neural = "checkpoint" in first
                    second_is_neural = "checkpoint" in second
                    if first_is_ngram and second_is_ngram:
                        if a.csv:
                            self.compare_ngram_csv(a.csv, first, second, a.n)
                        else:
                            self.compare_ngram_models(a.test, first, second, a.limit, a.print_progress)
                    elif first_is_ngram and second_is_neural:
                        if a.csv:
                            self.compare_csv(a.csv, first, second, a.n)
                        else:
                            self.evaluate(a.test, first, second, a.limit, a.print_progress)
                    elif first_is_neural and second_is_ngram:
                        if a.csv:
                            self.compare_csv(a.csv, second, first, a.n)
                        else:
                            self.evaluate(a.test, second, first, a.limit, a.print_progress)
                    else:
                        raise ValueError("compare supports one n-gram and one neural model, or two n-gram models")
                else:
                    print("Unknown command; enter help for usage.")
            except SystemExit:
                continue
            except (ValueError, FileNotFoundError, KeyError) as exc:
                print(f"error: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Interactive language-model evaluation shell.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()
    LanguageModelCLI(args.device).shell()


if __name__ == "__main__":
    main()
