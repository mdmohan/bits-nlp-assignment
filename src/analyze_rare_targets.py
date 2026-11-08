"""Exhaustive held-out rare-word and rare-combination results from verified scores."""

import argparse
import csv
import gzip
import html
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_inaugural import (
    MODELS, aggregate, contained, make_events, read_json, summarize_losses,
    verified_inputs, write_csv,
)
from lm.data import BOS, EOS, Vocab
from lm.evaluation import original_counts
from lm.preparation import atomic_json, file_sha256, fingerprint, sentences_for


RARE_MAX = 5


def rarity(count):
    if count < 0:
        raise ValueError("A training count cannot be negative")
    return "unseen_0" if count == 0 else "rare_1_5" if count <= RARE_MAX else "common_6_plus"


def build_inventory(events, word_counts, gram_counts, vocab):
    lexical_occurrences = defaultdict(list)
    combination_occurrences = defaultdict(list)
    groups = defaultdict(list)
    test_words = Counter()
    test_grams = Counter()
    for event in events:
        if event["eos"]:
            continue
        i, target = event["event_index"], event["target"]
        gram = tuple(event["context"] + [target])
        word_rarity = rarity(word_counts[target])
        gram_rarity = rarity(gram_counts[gram])
        test_words[target] += 1
        test_grams[gram] += 1
        groups["all_word_targets"].append(i)
        groups["word:" + word_rarity].append(i)
        groups["trigram:" + gram_rarity].append(i)
        groups["joint:" + word_rarity + ":" + gram_rarity].append(i)
        known_target = target in vocab
        groups["target:" + ("known" if known_target else "oov")].append(i)
        if word_counts[target] <= RARE_MAX:
            lexical_occurrences[target].append(i)
            groups["word:rare_or_unseen"].append(i)
            groups["word:" + word_rarity + ":" + ("known" if known_target else "oov")].append(i)
        if gram_counts[gram] <= RARE_MAX:
            combination_occurrences[gram].append(i)
            groups["trigram:rare_or_unseen"].append(i)
            if all(word in vocab for word in gram):
                groups["trigram:" + gram_rarity + ":all_words_known"].append(i)
        if word_counts[target] <= RARE_MAX or gram_counts[gram] <= RARE_MAX:
            groups["eligible_union"].append(i)
        if word_counts[target] <= RARE_MAX and gram_counts[gram] <= RARE_MAX:
            groups["eligible_intersection"].append(i)
    for dimension in ("word", "trigram"):
        for value in ("unseen_0", "rare_1_5", "common_6_plus"):
            groups.setdefault(dimension + ":" + value, [])
    for value in ("all_word_targets", "eligible_union", "eligible_intersection", "target:known", "target:oov",
                  "word:rare_or_unseen", "trigram:rare_or_unseen"):
        groups.setdefault(value, [])
    lexical_inventory = []
    for target in sorted(set(word_counts) | set(test_words)):
        count = word_counts[target]
        if count <= RARE_MAX:
            lexical_inventory.append({
                "target": target, "training_count": count, "rarity": rarity(count),
                "mapped_target": vocab.itos[vocab.stoi.get(target, vocab.unk_id)],
                "target_known": target in vocab, "heldout_occurrences": test_words[target],
                "scored_occurrences": len(lexical_occurrences[target]),
                "evaluation_status": "scored_all_heldout_occurrences" if test_words[target] else "no_heldout_context",
            })
    combination_inventory = []
    for gram in sorted(combination_occurrences):
        mapped = [vocab.itos[vocab.stoi.get(word, vocab.unk_id)] for word in gram]
        combination_inventory.append({
            "context_1": gram[0], "context_2": gram[1], "target": gram[2],
            "training_count": gram_counts[gram], "rarity": rarity(gram_counts[gram]),
            "heldout_occurrences": test_grams[gram], "all_words_known": all(word in vocab for word in gram),
            "mapped_context_1": mapped[0], "mapped_context_2": mapped[1], "mapped_target": mapped[2],
        })
    return lexical_inventory, combination_inventory, lexical_occurrences, combination_occurrences, groups


def event_metrics(logp, predictions, targets, known, indices, total):
    result = summarize_losses(logp, predictions, targets, indices, total)
    selected = np.asarray(indices, dtype=int)
    lexical_correct = int(((predictions[selected] == targets[selected]) & known[selected]).sum())
    result["lexical_correct"] = lexical_correct
    result["lexical_accuracy"] = lexical_correct / len(selected) if len(selected) else None
    return result


def aggregate_rows(rows):
    summary = aggregate(rows)
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["budget_percent"], row["model"], row["group"])].append(row)
    for row in summary:
        values = [r["lexical_accuracy"] for r in by_key[(row["budget_percent"], row["model"], row["group"])]]
        row["lexical_accuracy_mean"] = None
        row["lexical_accuracy_std"] = None
        if all(value is not None for value in values):
            row["lexical_accuracy_mean"] = statistics.mean(values)
            row["lexical_accuracy_std"] = statistics.stdev(values) if len(values) > 1 else None
    return summary


def paired_results(budget, groups, scores, targets, known):
    left = [scores[("lstm", seed)] for seed in (42, 43, 44)]
    right = [scores[("ffnn", seed)] for seed in (42, 43, 44)]
    advantage = np.mean([x[0] for x in left], axis=0) - np.mean([x[0] for x in right], axis=0)
    left_wrong = np.all([prediction != targets for _, prediction in left], axis=0)
    right_wrong = np.all([prediction != targets for _, prediction in right], axis=0)
    rows = []
    for group, indices in groups.items():
        ids = np.asarray(indices, dtype=int)
        delta = advantage[ids]
        rows.append({
            "budget_percent": budget, "group": group, "events": len(ids),
            "lstm_probability_wins": int((delta > 0).sum()),
            "ffnn_probability_wins": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
            "lstm_mean_log_probability_advantage_nats": float(delta.mean()) if len(ids) else None,
            "both_mapped_wrong_all_six": int((left_wrong[ids] & right_wrong[ids]).sum()),
            "both_lexically_wrong_all_six": int(((left_wrong[ids] | ~known[ids]) &
                                                (right_wrong[ids] | ~known[ids])).sum()),
            "known_target_events": int(known[ids].sum()),
        })
    return rows


def type_results(budget, unit, occurrences, word_counts, gram_counts, scores, targets, known, total):
    for value, indices in sorted(occurrences.items()):
        if not indices:
            continue
        label = value if unit == "word" else json.dumps(value, separators=(",", ":"))
        raw_count = word_counts[value] if unit == "word" else gram_counts[value]
        rows = [
            dict(budget_percent=budget, model=model, seed=seed, group=label,
                 **event_metrics(logp, predictions, targets, known, indices, total))
            for (model, seed), (logp, predictions) in scores.items()
        ]
        for record in aggregate_rows(rows):
            record.update(unit=unit, training_count=raw_count, rarity=rarity(raw_count))
            yield record


def render_report(output, inventory, summary, comparisons, coverage, verification):
    def table(rows, columns):
        header = "<tr>" + "".join("<th>" + html.escape(c) + "</th>" for c in columns) + "</tr>"
        body = []
        for row in rows:
            values = [f"{row[c]:.6g}" if isinstance(row[c], float) else "" if row[c] is None else str(row[c])
                      for c in columns]
            body.append("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in values) + "</tr>")
        return "<table>" + header + "".join(body) + "</table>"

    style = "body{font:15px system-ui;max-width:1400px;margin:25px auto;padding:15px}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:5px 8px;border:1px solid #ddd}pre{white-space:pre-wrap}input{font-size:16px;padding:8px;width:70%}"
    links = (
        "<ul><li><a href='lexical_inventory.csv'>Complete rare-word inventory</a></li>"
        "<li><a href='combination_inventory.csv'>Complete held-out rare/unseen trigram inventory</a></li>"
        "<li><a href='target_results.csv.gz'>Results for every scorable rare word, all models</a></li>"
        "<li><a href='combination_results.csv.gz'>Results for every rare/unseen trigram, all models</a></li>"
        "<li><a href='event_metadata.csv.gz'>Every eligible original-context occurrence</a></li>"
        "<li><a href='event_scores.csv.gz'>Every eligible occurrence's model/seed probability and prediction</a></li>"
        "<li><a href='summary.csv'>Seed-aggregate metrics</a></li><li><a href='runs.csv'>Every seed's metrics</a></li>"
        "<li><a href='paired_comparisons.csv'>Paired neural probability wins and shared failures</a></li></ul>"
    )
    sections = [
        "<h1>Exhaustive rare targets: every held-out occurrence</h1>",
        "<p>No example cap, candidate sampling, sentence subsampling or top-k filtering. "
        "Original normalized word frequency and original trigram frequency are counted separately from each budget's "
        "training documents. Unseen means count zero; rare means count 1-5. EOS is excluded. "
        "All three budgets and all three neural seeds are retained.</p>",
        "<p>Results use the previously verified probability of every original test event under every frozen checkpoint. "
        "This is exact extraction of already executed model inference, not another training or inference run. "
        "Full LSTM prefixes and context-2 FFNN inputs are preserved; isolated word lists were not substituted as prompts.</p>",
        "<p>The lexical inventory also lists training-rare words that never occur in test. These have "
        "<code>no_heldout_context</code>, zero scorable occurrences and no invented test score. "
        "Unseen words are limited to words actually observed in held-out data, not an enumerable universe of possible strings. "
        "The combination inventory contains every matching trigram type observed in held-out word events.</p>",
        "<p>Mapped accuracy includes correct UNK predictions. Lexical accuracy gives no credit for predicting UNK "
        "instead of the original word. PPL/NLL for OOV events still concern the mapped target, not original-word recovery. "
        "Zero-probability MLE events remain infinite, with explicit status and blank numerical loss fields.</p>",
        "<h2>Complete coverage</h2>",
        table(coverage, list(coverage[0])),
        "<h2>Results, mean and sample SD over neural seeds</h2>",
        "<p>Each metric is calculated over every target occurrence in that row. Word and trigram groups overlap: "
        "do not add them together. Within either dimension, unseen/rare/common partition all 25,821 word events. "
        "Vocabulary growth changes target mapping and output support across budgets. "
        "Sample SD measures initialization variation, not dataset uncertainty.</p>",
        table([r for r in summary if r["group"] in {
            "word:unseen_0", "word:rare_1_5", "word:rare_1_5:known", "word:rare_1_5:oov",
            "trigram:unseen_0", "trigram:rare_1_5", "trigram:unseen_0:all_words_known",
        }], ["budget_percent", "model", "group", "events", "seeds", "nll_mean", "nll_std",
             "perplexity_mean", "accuracy_mean", "lexical_accuracy_mean", "zero_probability_events_mean", "status"]),
        "<h2>Paired neural comparison</h2>",
        "<p>A probability win means higher mean target log probability across seeds 42/43/44, not necessarily "
        "correct top-1 prediction. These are exhaustive descriptive counts, not significance tests.</p>",
        table([r for r in comparisons if r["group"] in {
            "word:unseen_0", "word:rare_1_5", "trigram:unseen_0", "trigram:rare_1_5",
        }], list(comparisons[0])),
        "<h2>Files</h2>", links, "<p><a href='rare_targets.html'>Searchable complete rare-word inventory</a></p>",
        "<h2>Verification</h2><pre>" + html.escape(json.dumps(verification, indent=2)) + "</pre>",
    ]
    (output / "report.html").write_text("<!doctype html><meta charset='utf-8'><title>Exhaustive rare-target results</title>"
                                       "<style>" + style + "</style>" + "\n".join(sections), encoding="utf-8")
    inventory_html = (
        "<!doctype html><meta charset='utf-8'><title>Complete rare-word inventory</title><style>" + style + "</style>"
        "<h1>Every rare training word and unseen held-out word</h1>"
        "<p>Count 1-5 in training is rare; count zero is unseen. All inventories are budget-specific. "
        "Words without held-out contexts are listed but cannot receive a held-out evaluation.</p>"
        "<label>Filter any column: <input id='filter' type='search'></label>"
        + table(inventory, list(inventory[0])) +
        "<script>document.getElementById('filter').addEventListener('input',function(){"
        "const q=this.value.toLowerCase();document.querySelectorAll('table tr:not(:first-child)').forEach("
        "r=>{r.hidden=!r.textContent.toLowerCase().includes(q);});});</script>"
    )
    (output / "rare_targets.html").write_text(inventory_html, encoding="utf-8")


def run(experiment, analysis, output):
    started = time.perf_counter()
    experiment, analysis, output = (Path(p).resolve() for p in (experiment, analysis, output))
    for source in (experiment, analysis):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Use a new output directory separate from both frozen inputs")
    output.mkdir(parents=True, exist_ok=False)
    cache_manifest = read_json(analysis / "artifact_manifest.json")
    cache_protocol = read_json(analysis / "protocol.json")
    if cache_manifest["protocol_fingerprint"] != fingerprint(cache_protocol):
        raise ValueError("Cached scoring protocol identity mismatch")
    for name, digest in cache_manifest["files"].items():
        if file_sha256(contained(analysis, name)) != digest:
            raise ValueError(f"Cached scoring artifact changed: {name}")
    config, prepared, seal, verified = verified_inputs(experiment, output, "inaugural")
    if (cache_protocol["experiment_seal"] != seal["fingerprint"] or
            cache_protocol["prepared_fingerprint"] != prepared["fingerprint"] or
            cache_protocol["input_bundle_sha256"] != file_sha256(experiment / "input_bundle.zip")):
        raise ValueError("Cached scores are from a different experiment or prepared population")
    budgets = config["budgets_percent"]
    if budgets != [10, 50, 100] or cache_protocol["neural_seeds"] != [42, 43, 44]:
        raise ValueError("Expected the frozen three-budget, three-seed experiment")
    vocabs = {b: Vocab.from_list(read_json(experiment / "inaugural" / f"budget_{b:03}" / "vocab.json")) for b in budgets}
    test = sentences_for(prepared, "test")
    events = make_events(test, prepared, vocabs, budgets)
    with (analysis / "events.csv").open(newline="", encoding="utf-8") as stream:
        cached_events = list(csv.DictReader(stream))
    if len(cached_events) != len(events):
        raise ValueError("Cached event population differs")
    for cached, current in zip(cached_events, events):
        for key in ("event_id", "target", "original_target", "source", "position", "sentence_position", "event_index"):
            if cached[key] != str(current[key]):
                raise ValueError(f"Cached original event alignment mismatch: {key}")
    prior_word_metrics = defaultdict(list)
    with (analysis / "decomposition.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["group"] in ("target:in_vocabulary", "target:oov"):
                prior_word_metrics[(int(row["budget_percent"]), row["model"], int(row["seed"]))].append(row)
    with np.load(analysis / "scores.npz", allow_pickle=False) as archive:
        expected = {
            f"b{b}_{m}_s{seed}_{kind}"
            for b in budgets for m in MODELS for seed in ((42, 43, 44) if m in ("lstm", "ffnn") else (0,))
            for kind in ("logp", "prediction_id")
        }
        if set(archive.files) != expected:
            raise ValueError("Incomplete or unexpected cached model/seed matrix")
        cached_scores = {key: archive[key] for key in expected}
    protocol = {
        "name": "inaugural_rare_targets_v1", "schema_version": 1, "rare_max_training_count": RARE_MAX,
        "training_performed": False, "new_inference_performed": False,
        "scoring_source": "verified exhaustive per-event inference from inaugural_analysis100_v1",
        "source_cache_protocol": fingerprint(cache_protocol), "source_experiment_seal": seal["fingerprint"],
        "script_sha256": file_sha256(__file__), "budgets_percent": budgets, "neural_seeds": [42, 43, 44],
        "normalization": "original words after lowercase/whitespace; rarity before UNK mapping",
        "population": "all held-out word occurrences; EOS excluded; no limits or sampling",
        "lexical_inventory": "all count-1-to-5 training word types plus observed count-zero held-out types; flag types without test context",
        "combination_inventory": "all held-out original two-word-context plus target types with training count 0-to-5",
        "context": "original full sentence prefix for LSTM; two preceding mapped tokens for FFNN; BOS handling unchanged",
        "word_probability": "P(mapped target|model-visible context), not recovery of an OOV original word",
        "comparison": "mean log probability across all three seeds; strict wins; no outcome-based filtering",
        "aggregation": "token-mean NLL and mapped/lexical accuracy; seed mean/sample SD; infinite MLE zeros explicit",
        "categories_overlap": True, "exploratory": True,
    }
    atomic_json(output / "protocol.json", protocol)
    all_inventory, all_combinations, all_metrics, all_pairs, coverage = [], [], [], [], []
    total_words = sum(not e["eos"] for e in events)
    score_record_count = 0
    with gzip.open(output / "event_metadata.csv.gz", "wt", newline="", encoding="utf-8") as context_stream, \
            gzip.open(output / "event_scores.csv.gz", "wt", newline="", encoding="utf-8") as score_stream, \
            gzip.open(output / "target_results.csv.gz", "wt", newline="", encoding="utf-8") as target_stream, \
            gzip.open(output / "combination_results.csv.gz", "wt", newline="", encoding="utf-8") as combination_stream:
        context_writer = score_writer = target_writer = combination_writer = None
        for b in budgets:
            vocab = vocabs[b]
            train = list(sentences_for(prepared, "train", b))
            words = Counter(w for sentence in train for w in sentence)
            grams = original_counts(train)[3]
            inventory, combinations, word_ids, gram_ids, groups = build_inventory(events, words, grams, vocab)
            all_inventory.extend(dict(budget_percent=b, **row) for row in inventory)
            all_combinations.extend(dict(budget_percent=b, **row) for row in combinations)
            targets = np.array([vocab.stoi.get(e["target"], vocab.unk_id) for e in events])
            known = np.array([e["target"] in vocab for e in events], dtype=bool)
            scores = {}
            for model in MODELS:
                for seed in ((42, 43, 44) if model in ("lstm", "ffnn") else (0,)):
                    logp = cached_scores[f"b{b}_{model}_s{seed}_logp"]
                    predictions = cached_scores[f"b{b}_{model}_s{seed}_prediction_id"]
                    if logp.shape != (len(events),) or predictions.shape != (len(events),):
                        raise ValueError("Cached score array has the wrong event shape")
                    if (np.isnan(logp).any() or np.isposinf(logp).any() or (logp > 1e-6).any() or
                            (predictions < -1).any() or (predictions >= len(vocab)).any()):
                        raise ValueError("Invalid cached scores or prediction IDs")
                    scores[(model, seed)] = (logp, predictions)
                    word_metrics = event_metrics(logp, predictions, targets, known, groups["all_word_targets"], total_words)
                    previous = prior_word_metrics[(b, model, seed)]
                    if (len(previous) != 2 or sum(int(r["events"]) for r in previous) != total_words or
                            sum(int(r["zero_probability_events"]) for r in previous) != word_metrics["zero_probability_events"] or
                            sum(int(r["correct"]) for r in previous) != word_metrics["correct"] or
                            not math.isclose(sum(float(r["finite_nll_sum"]) for r in previous),
                                             word_metrics["finite_nll_sum"], rel_tol=1e-12, abs_tol=1e-8)):
                        raise ValueError("Cached word-only loss/correctness differs from the verified prior analysis")
                    for group, indices in groups.items():
                        all_metrics.append(dict(budget_percent=b, model=model, seed=seed, group=group,
                                                **event_metrics(logp, predictions, targets, known, indices, total_words)))
                    for i in groups["eligible_union"]:
                        e = events[i]
                        prediction = None if predictions[i] < 0 else vocab.itos[predictions[i]]
                        mapped = vocab.itos[targets[i]]
                        row = {
                            "budget_percent": b, "event_id": e["event_id"], "model": model, "seed": seed,
                            "original_target": e["original_target"], "normalized_target": e["target"],
                            "mapped_target": mapped, "target_log_probability": None if np.isneginf(logp[i]) else float(logp[i]),
                            "target_probability": math.exp(logp[i]),
                            "probability_status": "zero_probability" if np.isneginf(logp[i]) else "finite",
                            "prediction": prediction, "mapped_correct": prediction == mapped,
                            "lexically_correct": prediction == e["target"],
                        }
                        if score_writer is None:
                            score_writer = csv.DictWriter(score_stream, fieldnames=list(row))
                            score_writer.writeheader()
                        score_writer.writerow(row)
                        score_record_count += 1
            for dimension in ("word", "trigram"):
                indices = [groups[dimension + ":" + label] for label in ("unseen_0", "rare_1_5", "common_6_plus")]
                if sorted(i for part in indices for i in part) != groups["all_word_targets"]:
                    raise ValueError("Rarity buckets do not partition the entire held-out word population")
            for i in groups["eligible_union"]:
                e = events[i]
                gram = tuple(e["context"] + [e["target"]])
                prefix = [BOS] + e["words"][:e["position"]]
                mapped_prefix = [vocab.itos[vocab.stoi.get(w, vocab.unk_id)] for w in prefix]
                row = {
                    "budget_percent": b, "event_id": e["event_id"], "source": e["source"],
                    "sentence_position": e["sentence_position"], "target_position": e["position"],
                    "original_sentence": e["sentence"], "original_target": e["original_target"],
                    "normalized_target": e["target"], "mapped_target": vocab.itos[targets[i]],
                    "word_training_count": words[e["target"]], "word_rarity": rarity(words[e["target"]]),
                    "trigram_training_count": grams[gram], "trigram_rarity": rarity(grams[gram]),
                    "original_context": json.dumps(e["context"]), "lstm_mapped_prefix": json.dumps(mapped_prefix),
                    "ffnn_mapped_context": json.dumps(([BOS] + mapped_prefix)[-2:]),
                }
                if context_writer is None:
                    context_writer = csv.DictWriter(context_stream, fieldnames=list(row))
                    context_writer.writeheader()
                context_writer.writerow(row)
            for unit, ids, stream in (("word", word_ids, target_stream), ("trigram", gram_ids, combination_stream)):
                for row in type_results(b, unit, ids, words, grams, scores, targets, known, total_words):
                    if unit == "word":
                        if target_writer is None:
                            target_writer = csv.DictWriter(stream, fieldnames=list(row))
                            target_writer.writeheader()
                        target_writer.writerow(row)
                    else:
                        if combination_writer is None:
                            combination_writer = csv.DictWriter(stream, fieldnames=list(row))
                            combination_writer.writeheader()
                        combination_writer.writerow(row)
            all_pairs.extend(paired_results(b, groups, scores, targets, known))
            coverage.append({
                "budget_percent": b, "word_events": total_words,
                "rare_training_word_types": sum(r["training_count"] > 0 for r in inventory),
                "rare_training_types_with_test_context": sum(r["training_count"] > 0 and r["heldout_occurrences"] > 0 for r in inventory),
                "rare_training_types_without_test_context": sum(r["training_count"] > 0 and not r["heldout_occurrences"] for r in inventory),
                "unseen_test_word_types": sum(r["training_count"] == 0 for r in inventory),
                "rare_word_events": len(groups["word:rare_1_5"]),
                "unseen_word_events": len(groups["word:unseen_0"]),
                "rare_trigram_types": sum(r["training_count"] > 0 for r in combinations),
                "unseen_trigram_types": sum(r["training_count"] == 0 for r in combinations),
                "rare_trigram_events": len(groups["trigram:rare_1_5"]),
                "unseen_trigram_events": len(groups["trigram:unseen_0"]),
                "eligible_union_events": len(groups["eligible_union"]),
                "eligible_intersection_events": len(groups["eligible_intersection"]),
                "score_records": len(groups["eligible_union"]) * len(scores),
            })
            print(f"Exhaustive budget {b}%: {coverage[-1]}", flush=True)
    summary = aggregate_rows(all_metrics)
    write_csv(output / "lexical_inventory.csv", all_inventory)
    write_csv(output / "combination_inventory.csv", all_combinations)
    write_csv(output / "runs.csv", all_metrics)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_comparisons.csv", all_pairs)
    write_csv(output / "coverage.csv", coverage)
    if score_record_count != sum(row["score_records"] for row in coverage):
        raise ValueError("Incomplete exhaustive score export")
    for name, digest in cache_manifest["files"].items():
        if file_sha256(analysis / name) != digest:
            raise ValueError("Source analysis changed during extraction")
    for name, digest in verified.items():
        if file_sha256(experiment / name) != digest:
            raise ValueError("Frozen experiment changed during extraction")
    verification = {
        "complete_heldout_word_population": total_words, "eos_excluded": len(test),
        "budgets": budgets, "scored_model_seed_runs": 27, "event_score_records": score_record_count,
        "lexical_inventory_rows": len(all_inventory), "combination_inventory_rows": len(all_combinations),
        "all_cached_events_aligned": True, "all_rarity_partitions_complete": True,
        "all_27_word_losses_and_correct_counts_match_prior_analysis": True,
        "all_source_artifacts_unchanged": True, "sampling_or_caps": False,
        "new_inference_performed": False, "training_performed": False,
        "cache_files_sha256_verified": len(cache_manifest["files"]), "sealed_files_verified": len(verified),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "verification.json", verification)
    render_report(output, all_inventory, summary, all_pairs, coverage, verification)
    atomic_json(output / "artifact_manifest.json", {
        "protocol_fingerprint": fingerprint(protocol),
        "files": {p.relative_to(output).as_posix(): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
    })
    print("EXHAUSTIVE_COMPLETE: " + str(output / "report.html"), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True, help="Complete verified analysis, including scores.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment, args.analysis, args.output)


if __name__ == "__main__":
    main()
