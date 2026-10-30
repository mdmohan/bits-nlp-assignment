"""Aligned, streaming next-token metrics for count and neural models."""

import math
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import BOS, EOS, mask_logits


def sentence_events(ids, order, bos, eos):
    padded = [bos] * (order - 1) + list(ids) + [eos]
    for position in range(len(ids) + 1):
        offset = position + order - 1
        yield tuple(padded[offset - order + 1:offset]), padded[offset]


def original_counts(sentences):
    counts = {2: Counter(), 3: Counter()}
    for words in sentences:
        for order in counts:
            for context, target in sentence_events(words, order, BOS, EOS):
                counts[order][context + (target,)] += 1
    return counts


def _event(words, ids, sentence_index, position, target, log_probability, prediction, covered=None):
    return {"sentence_index": sentence_index, "position": position, "target": target,
            "log_probability": log_probability, "prediction": prediction, "covered": covered,
            "words": words, "ids": ids, "eos": position == len(words)}


def ngram_events(model, vocab, sentences):
    for sentence_index, words in enumerate(sentences):
        ids = vocab.encode(words)[1:-1]
        for position, (context, target) in enumerate(sentence_events(ids, model.order, vocab.stoi[BOS], vocab.stoi[EOS])):
            probability, prediction = model.score(context, target)
            log_probability = -math.inf
            if probability > 0:
                log_probability = math.log(probability)
            covered = bool(model.model.context_totals[model.order][context])
            yield _event(words, ids, sentence_index, position, target, log_probability, prediction, covered)


@torch.no_grad()
def neural_events(model, vocab, sentences, device, batch_size=32, max_padded_tokens=2048,
                  allowed_target_ids="vocab"):
    if allowed_target_ids == "vocab":
        allowed_target_ids = vocab.allowed_target_ids
    model.eval()

    def score(batch):
        encoded = [torch.tensor(vocab.encode(words), dtype=torch.long) for _, words in batch]
        inputs = pad_sequence([ids[:-1] for ids in encoded], batch_first=True, padding_value=vocab.pad_id).to(device)
        targets = pad_sequence([ids[1:] for ids in encoded], batch_first=True, padding_value=vocab.pad_id).to(device)
        logits = mask_logits(model(inputs).float(), allowed_target_ids)
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu()
        predictions = logits.argmax(-1).cpu()
        for row, ((sentence_index, words), ids) in enumerate(zip(batch, encoded)):
            content_ids = ids.tolist()[1:-1]
            for position, target in enumerate(ids.tolist()[1:]):
                yield _event(words, content_ids, sentence_index, position, target,
                             gathered[row, position].item(), predictions[row, position].item())

    batch = []
    longest = 0
    for sentence_index, words in enumerate(sentences):
        length = len(words) + 1
        if length > max_padded_tokens:
            raise ValueError("A sentence exceeds the padded-token resource limit; increase the limit, do not truncate")
        proposed = max(longest, length)
        if batch and (len(batch) >= batch_size or proposed * (len(batch) + 1) > max_padded_tokens):
            yield from score(batch)
            batch = []
            longest = 0
        batch.append((sentence_index, words))
        longest = max(longest, length)
    if batch:
        yield from score(batch)


def _totals():
    return dict(tokens=0, finite_nll_sum=0.0, zero_probability_events=0, correct=0,
                context_covered=0, correct_when_context_seen=0)


def _add(totals, event):
    log_probability = event["log_probability"]
    if math.isnan(log_probability) or log_probability > 1e-6:
        raise ValueError("Model produced an invalid log probability")
    totals["tokens"] += 1
    if log_probability == -math.inf:
        totals["zero_probability_events"] += 1
    else:
        totals["finite_nll_sum"] -= log_probability
    correct = event["prediction"] == event["target"]
    totals["correct"] += int(correct)
    totals["context_covered"] += int(event["covered"] is True)
    totals["correct_when_context_seen"] += int(correct and event["covered"] is True)


def _finish(totals):
    result = dict(totals, nll_sum=None, nll=None, perplexity=None, status="empty", accuracy=None)
    if not totals["tokens"]:
        return result
    result["accuracy"] = totals["correct"] / totals["tokens"]
    if totals["zero_probability_events"]:
        result["status"] = "infinite_zero_probability"
        return result
    result["nll_sum"] = totals["finite_nll_sum"]
    result["nll"] = result["nll_sum"] / totals["tokens"]
    result["status"] = "overflow"
    if result["nll"] < math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        result["perplexity"] = math.exp(result["nll"])
        result["status"] = "finite"
    return result


def evaluate_model(model, vocab, sentences, family, device=None, batch_size=32,
                   max_padded_tokens=2048, allowed_target_ids="vocab", event_sink=None,
                   mapped_reference=None, current_original=None, fixed_original=None, case_limit=10):
    totals = _totals()
    word_totals = _totals()
    groups = {}
    word_tokens = oov_tokens = sentence_count = 0
    cases = []
    if family == "ngram":
        events = ngram_events(model, vocab, sentences)
    elif family == "neural":
        events = neural_events(model, vocab, sentences, device, batch_size, max_padded_tokens, allowed_target_ids)
    else:
        raise ValueError(f"Unknown evaluation family: {family}")
    for event in events:
        _add(totals, event)
        sentence_count += int(event["eos"])
        is_oov = False
        if not event["eos"]:
            _add(word_totals, event)
            word_tokens += 1
            is_oov = event["words"][event["position"]] not in vocab
            oov_tokens += int(is_oov)
        label = "in_vocabulary"
        if event["eos"]:
            label = "eos"
        elif is_oov:
            label = "oov"
        labels = ["target:" + label]
        if mapped_reference is not None:
            prefix = [vocab.stoi[BOS]] * 2 + event["ids"][:event["position"]]
            gram = tuple(prefix[-2:]) + (event["target"],)
            frequency = mapped_reference.ngram_counts[1][(event["target"],)]
            bucket = "21+"
            if frequency == 0:
                bucket = "zero"
            elif frequency == 1:
                bucket = "1"
            elif frequency <= 5:
                bucket = "2-5"
            elif frequency <= 20:
                bucket = "6-20"
            event["mapped_trigram_count"] = mapped_reference.ngram_counts[3][gram]
            event["mapped_context_count"] = mapped_reference.context_totals[3][gram[:-1]]
            labels.append("training_frequency:" + bucket)
            for name, count in (("mapped_trigram", event["mapped_trigram_count"]), ("mapped_context", event["mapped_context_count"])):
                state = "unseen"
                if count:
                    state = "seen"
                labels.append(name + ":" + state)
        for name, reference in (("current_original", current_original), ("fixed_original", fixed_original)):
            if reference is not None:
                prefix = [BOS, BOS] + event["words"][:event["position"]]
                raw_target = EOS
                if not event["eos"]:
                    raw_target = event["words"][event["position"]]
                state = "unseen"
                if reference[3][tuple(prefix[-2:]) + (raw_target,)]:
                    state = "seen"
                labels.append(name + "_trigram:" + state)
        for label in labels:
            _add(groups.setdefault(label, _totals()), event)
        if event_sink is not None:
            event_sink(event)
        if not event["eos"] and len(cases) < case_limit:
            prediction = None
            if event["prediction"] is not None:
                prediction = vocab.itos[event["prediction"]]
            log_probability = event["log_probability"]
            if not math.isfinite(log_probability):
                log_probability = None
            raw_target = EOS
            if not event["eos"]:
                raw_target = event["words"][event["position"]]
            cases.append({"sentence_index": event["sentence_index"], "position": event["position"],
                          "full_prefix": [BOS] + event["words"][:event["position"]],
                          "raw_target": raw_target, "mapped_target": vocab.itos[event["target"]],
                          "prediction": prediction, "log_probability": log_probability,
                          "zero_probability": event["log_probability"] == -math.inf,
                          "mapped_trigram_count": event.get("mapped_trigram_count"),
                          "mapped_context_count": event.get("mapped_context_count")})
    result = _finish(totals)
    all_event_metrics = result.copy()
    word_metrics = _finish(word_totals)
    # Keep sentence-level likelihood metrics over all targets, including </s>,
    # while making the canonical prediction metrics word-target-only.
    result.update(
        accuracy=word_metrics["accuracy"],
        correct=word_metrics["correct"],
        context_covered=word_metrics["context_covered"],
        correct_when_context_seen=word_metrics["correct_when_context_seen"],
        all_event_accuracy=all_event_metrics["accuracy"],
        all_event_correct=all_event_metrics["correct"],
        all_event_context_covered=all_event_metrics["context_covered"],
    )
    result.update(sentences=sentence_count, word_tokens=word_tokens, oov_tokens=oov_tokens,
                  sentence_end_targets=sentence_count, word_metrics=word_metrics,
                  oov_rate=0.0, groups={name: _finish(values) for name, values in groups.items()}, cases=cases)
    if word_tokens:
        result["oov_rate"] = oov_tokens / word_tokens
    return result
