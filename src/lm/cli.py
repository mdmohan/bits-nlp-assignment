"""Interactive shell for loading and evaluating language models."""

import argparse
import heapq
import json
import math
import re
import shlex
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm.data import BOS, EOS, PAD, UNK, Vocab, mask_logits
from lm.evaluation import evaluate_model, sentence_events
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


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


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
        train_path = Path(data.get("output", "data/processed")) / data.get("corpus", "") / "train.txt"
        if not train_path.is_absolute():
            train_path = Path(__file__).resolve().parents[2] / train_path
        info = {
            "name": name, "model": model, "checkpoint": path, "vocab": vocab, "stoi": stoi,
            "run_dir": path.parent,
            "lowercase": data.get("lowercase", True),
            "train_path": train_path.resolve(),
            "train_percent": float(config.get("train_percent", 100)),
        }
        if "allowed_target_ids" in config:
            info["allowed_target_ids"] = config["allowed_target_ids"]
        key = str(path)
        self.neural[key] = info
        self.active_neural = key
        print(f"Loaded neural {name}: {path.name} (vocab {len(vocab):,}, device {self.device})")
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
        words = text.split()
        if info["lowercase"]:
            words = [w.lower() for w in words]
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
        """Read line-based corpora, splitting packed prose lines only as needed."""
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sentences = _SENTENCE_BOUNDARY.split(line) if any(
                    marker in line for marker in (". ", "? ", "! ")
                ) else (line,)
                for sentence in sentences:
                    words = sentence.split()
                    if words:
                        yield words

    def load_ngram(
        self,
        name: str,
        order: int,
        train_path: Path,
        min_freq: int = 2,
        smoothing: str = "mle",
        k: float | None = None,
        discount: float = 0.75,
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
        lowercase = nn_info["lowercase"] if nn_info else True
        train_percent = nn_info["train_percent"] if nn_info else 100
        if nn_info:
            vocab = nn_info["vocab"]
        else:
            counts = Counter()
            total = 0
            for words in self._read_sentences(train_path):
                words = [w.lower() for w in words] if lowercase else words
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
            f"{self._display_path(train_path)}",
            flush=True,
        )
        started_at = time.perf_counter()
        allowed_target_ids = None
        if nn_info:
            allowed_target_ids = nn_info.get("allowed_target_ids")
        lm = CountNGram(
            order,
            vocab,
            iter_sentences(train_path, lowercase, train_percent),
            smoothing=smoothing,
            k=1.0 if k is None else k,
            discount=discount,
            allowed_target_ids=allowed_target_ids,
        )
        self.ngrams[name] = {
            "name": name, "order": order, "lm": lm, "vocab": vocab,
            "stoi": {w: i for i, w in enumerate(vocab)}, "lowercase": lowercase,
            "train_path": train_path,
            "smoothing": smoothing, "k": k, "discount": discount,
        }
        stats = [
            ("Status", "ready in this session"),
            ("Name", name),
            ("Order", str(order)),
            ("Smoothing", smoothing_label),
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

        def retain(event, family):
            nonlocal last_progress_at
            row_index = counters[family]
            counters[family] += 1
            now = time.perf_counter()
            if row_index % 10_000 == 0 and progress_seconds and now - last_progress_at >= progress_seconds:
                print(f"  evaluation progress ({family}): {row_index:,} events ({now - started_at:.1f}s elapsed)", flush=True)
                last_progress_at = now
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
            record = {
                "index": row_index,
                "key": event_key,
                "context": " ".join(vocab[token] for token in context),
                "expected": vocab[event["target"]],
                "prediction_text": vocab[prediction] if prediction is not None else "<unseen>",
                "expected_probability": probability,
                "prediction_probability": ngram_info["lm"].prediction_probability(context),
                "correct": prediction == event["target"],
            }
            samples = correct_samples if record["correct"] else incorrect_samples
            heap = correct_heap if record["correct"] else incorrect_heap
            if limit == 0:
                samples[row_index] = record
                selected_by_event[event_key] = record
            elif len(heap) < limit:
                heapq.heappush(heap, (record["prediction_probability"], -row_index, row_index, record))
                samples[row_index] = record
                selected_by_event[event_key] = record
            elif (record["prediction_probability"], -row_index) > heap[0][:2]:
                removed = heapq.heapreplace(
                    heap,
                    (record["prediction_probability"], -row_index, row_index, record),
                )
                samples.pop(removed[2], None)
                selected_by_event.pop(removed[3]["key"], None)
                samples[row_index] = record
                selected_by_event[event_key] = record

        ng_result = nn_result = None
        ng_nll = nn_nll = 0.0
        ng_zero = ng_correct = ng_covered = nn_correct = 0
        if ngram_info:
            ng_result = evaluate_model(
                ngram_info["lm"], metric_vocab, iter_sentences(path, lowercase), "ngram",
                event_sink=lambda event: retain(event, "ngram"), case_limit=0,
            )
            ng_nll = ng_result["finite_nll_sum"]
            ng_zero = ng_result["zero_probability_events"]
            ng_correct = ng_result["correct"]
            ng_covered = ng_result["context_covered"]
        if neural_info:
            nn_result = evaluate_model(
                neural_info["model"], metric_vocab, iter_sentences(path, lowercase), "neural", self.device,
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
            f"  Prediction events: {total:,} "
            f"({words_count:,} word targets + {sentence_count:,} sentence-ending targets)"
        )
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
            accuracy = ng_correct / total if total else math.nan
            seen_accuracy = math.nan
            if ng_covered:
                seen_accuracy = ng_result["correct_when_context_seen"] / ng_covered
            context_coverage = (
                f"{ng_covered:,} / {total:,} ({ng_covered / total:.4%})" if total else "n/a"
            )
            model_results.append((
                f"{ngram_info['name']} ({order}-gram, {smoothing_desc})",
                [
                    ("Correct predictions", f"{ng_correct:,} / {total:,} ({accuracy:.2%})"),
                    ("Top-1 accuracy (all)", f"{accuracy:.2%}"),
                    ("Top-1 when context seen", f"{seen_accuracy:.2%}" if ng_covered else "n/a"),
                    ("Contexts seen in training", context_coverage),
                    ("Zero-probability targets", f"{ng_zero:,} / {total:,} ({ng_zero / total:.2%})" if total else "n/a"),
                    ("Perplexity", perplexity_text(ng_result)),
                ] if total else [("Evaluation", "No events")],
            ))
        if neural_info:
            accuracy = nn_correct / total if total else math.nan
            nn_zero = nn_result["zero_probability_events"]
            model_results.append((
                f"{neural_info['name']} ({neural_info['checkpoint'].name})",
                [
                    ("Correct predictions", f"{nn_correct:,} / {total:,} ({accuracy:.2%})"),
                    ("Top-1 accuracy (all)", f"{accuracy:.2%}"),
                    ("Top-1 when context seen", "n/a (neural model)"),
                    ("Contexts seen in training", "n/a (neural model)"),
                    ("Zero-probability targets", f"{nn_zero:,} / {total:,} ({nn_zero / total:.2%})"),
                    ("Perplexity", perplexity_text(nn_result)),
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

        print("\nTop-1 means the model's most likely next token matched the expected token.")
        print("Context coverage counts events whose n-gram context appeared in training.")
        if ngram_info and ng_zero and ngram_info["smoothing"] == "mle":
            print("An unsmoothed n-gram assigns zero probability to some targets, making perplexity infinite.")
        print(
            "Each sentence-ending target is </s>: it scores the model's probability of ending "
            "the sentence after its final word."
        )
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
                shown = "all" if limit == 0 else f"top {limit}"
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
        print("  ngram NAME --n N --train TRAIN.txt [--min-freq N] [--smoothing mle|laplace|backoff]")
        print("       Laplace requires --k; backoff accepts --discount (default 0.75).")
        print("  ngram NAME --evaluate FILE.txt [--limit N] [--print-progress SECONDS]")
        print("       Shows up to 25 correct and incorrect n-gram predictions, ranked by P(prediction); --limit 0 shows all.")
        print("  compare --one ngram NAME --two neural CHECKPOINT --test FILE.txt [--limit N] [--print-progress SECONDS]")
        print("  models | help | quit")

    def shell(self):
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
                    a = parser.parse_args(args[1:])
                    if a.n is not None:
                        if a.train is None:
                            raise ValueError("--n requires --train")
                        self.load_ngram(
                            a.name, a.n, a.train, a.min_freq,
                            smoothing=a.smoothing, k=a.k, discount=a.discount,
                        )
                    else:
                        if a.name not in self.ngrams:
                            raise ValueError(f"Unknown n-gram: {a.name}")
                        self.evaluate(a.evaluate, self.ngrams[a.name], None, a.limit, a.print_progress)
                elif cmd == "compare":
                    parser = argparse.ArgumentParser(prog="compare", add_help=False)
                    parser.add_argument("--one", nargs=2, required=True, metavar=("TYPE", "NAME"))
                    parser.add_argument("--two", nargs=2, required=True, metavar=("TYPE", "NAME"))
                    parser.add_argument("--test", type=Path, required=True)
                    parser.add_argument("--limit", type=int, default=25)
                    parser.add_argument("--print-progress", type=float, default=10.0)
                    a = parser.parse_args(args[1:])
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
                    ng = next((m for m in models if "order" in m), None)
                    nn_info = next((m for m in models if "checkpoint" in m), None)
                    if len(models) != 2 or ng is None or nn_info is None:
                        raise ValueError("compare currently requires one ngram and one neural model")
                    self.evaluate(a.test, ng, nn_info, a.limit, a.print_progress)
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
