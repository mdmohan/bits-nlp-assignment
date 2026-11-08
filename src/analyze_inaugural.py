"""Score frozen language models for stratified cases and stable-event loss analysis."""

import argparse
import csv
import hashlib
import html
import json
import math
import pickle
import platform
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from lm.data import BOS, EOS, Vocab
from lm.evaluation import neural_events, ngram_events, original_counts
from lm.neural import build_model
from lm.neural.trainer import get_device
from lm.preparation import atomic_json, file_sha256, fingerprint, readonly, sentences_for, verify_prepared_manifest


MODELS = ("bigram_mle", "trigram_mle", "trigram_addk", "lstm", "ffnn")
CATEGORIES = (
    ("known_word_unseen_combination", 30),
    ("oov_target", 30),
    ("ffnn_wins", 20),
    ("shared_failure", 20),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No records to write: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe artifact path: {name}")
    return path


def verified_inputs(experiment, output, corpus):
    seal = read_json(experiment / "seal.json")
    if seal["fingerprint"] != fingerprint(seal["content"]):
        raise ValueError("Experiment seal identity mismatch")
    verified = {}
    for relative, expected in seal["content"]["runs"].items():
        if isinstance(expected, dict):
            status = read_json(contained(experiment, relative + "/status.json"))
            if status["status"] != "complete" or status["artifacts"] != expected:
                raise ValueError(f"Sealed run status mismatch: {relative}")
            entries = [(relative + "/" + name, digest) for name, digest in expected.items()]
        else:
            entries = [(relative, expected)]
        for name, digest in entries:
            if file_sha256(contained(experiment, name)) != digest:
                raise ValueError(f"Sealed artifact changed: {name}")
            verified[name] = digest
    for name, expected in seal["content"]["prepared_manifests"].items():
        path = experiment / "prepared" / name / "manifest.json"
        if file_sha256(path) != expected:
            raise ValueError(f"Prepared manifest changed: {name}")
        verified[str(path.relative_to(experiment))] = expected
    bundle = experiment / "input_bundle.zip"
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("bundle_manifest.json"))
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Input bundle member changed: {name}")
        config = yaml.safe_load(archive.read("input/suite.yaml"))
        prepared = json.loads(archive.read(f"input/prepared/{corpus}/manifest.json"))
        index = output / "inputs" / "sentences.sqlite"
        index.parent.mkdir()
        index.write_bytes(archive.read(f"input/prepared/{corpus}/sentences.sqlite"))
    prepared["index"] = str(index)
    verify_prepared_manifest(prepared)
    archived = read_json(experiment / "prepared" / corpus / "manifest.json")
    if prepared["fingerprint"] != archived["fingerprint"]:
        raise ValueError("Bundled data differs from sealed experiment preparation")
    return config, prepared, seal, verified


def make_events(test, prepared, vocabs, budgets):
    records = []
    with readonly(prepared["index"]) as connection:
        for sentence_index, row_id in enumerate(test.row_ids):
            document, source, sentence_position, original = connection.execute(
                """SELECT s.document,d.source_key,s.position,s.canonical
                   FROM sentences s JOIN documents d ON d.id=s.document WHERE s.rowid=?""",
                (row_id,),
            ).fetchone()
            words = test[sentence_index]
            original_words = original.split()
            for position in range(len(words) + 1):
                eos = position == len(words)
                target = EOS if eos else words[position]
                context = ([BOS, BOS] + words[:position])[-2:]
                known = [target in vocabs[b] for b in budgets]
                sequence_known = [all(w in vocabs[b] for w in context + [target]) for b in budgets]
                records.append({
                    "event_index": len(records), "event_id": f"{document}:{sentence_position}:{position}",
                    "document": document, "source": Path(source).name, "sentence_index": sentence_index,
                    "sentence_position": sentence_position, "position": position, "eos": eos,
                    "original_target": EOS if eos else original_words[position], "target": target,
                    "context": context, "sentence": original, "words": words,
                    "target_cohort": "eos" if eos else coverage_cohort(known, budgets),
                    "sequence_cohort": "eos" if eos else coverage_cohort(sequence_known, budgets),
                })
    return records


def coverage_cohort(known, budgets):
    if all(known):
        return "known_all"
    if not any(known):
        return "oov_all"
    if any(left and not right for left, right in zip(known, known[1:])):
        raise ValueError("Vocabulary membership is not nested across budgets")
    return f"enters_at_{budgets[known.index(True)]}"


def summarize_losses(logp, predictions, targets, indices, total):
    selected = np.asarray(indices, dtype=int)
    values = logp[selected]
    zero = int(np.isneginf(values).sum())
    if np.isnan(values).any() or np.isposinf(values).any() or (values > 1e-6).any():
        raise ValueError("Invalid event log probabilities")
    finite_sum = float(-values[np.isfinite(values)].sum())
    nll = None if zero or not len(values) else finite_sum / len(values)
    ppl = None
    status = "empty" if not len(values) else "infinite_zero_probability" if zero else "finite"
    if nll is not None:
        if nll >= math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            status = "overflow"
        else:
            ppl = math.exp(nll)
    correct = int((predictions[selected] == targets[selected]).sum())
    return {
        "events": len(values), "event_fraction": len(values) / total,
        "zero_probability_events": zero, "finite_nll_sum": finite_sum,
        "nll_sum": None if zero else finite_sum, "nll": nll, "perplexity": ppl, "status": status,
        "nll_contribution": None if zero else finite_sum / total,
        "correct": correct, "accuracy": correct / len(values) if len(values) else None,
    }


def group_indices(events, vocab, raw_counts, reference):
    groups = defaultdict(list)
    for event in events:
        index = event["event_index"]
        target = event["target"]
        context = event["context"]
        target_type = "eos" if event["eos"] else "in_vocabulary" if target in vocab else "oov"
        groups["overall"].append(index)
        groups["target:" + target_type].append(index)
        groups["target_cohort:" + event["target_cohort"]].append(index)
        groups["sequence_cohort:" + event["sequence_cohort"]].append(index)
        count = raw_counts[3][tuple(context + [target])]
        rarity = "unseen" if count == 0 else "rare_1_5" if count <= 5 else "common_6_plus"
        groups["original_trigram:" + rarity].append(index)
        mapped = tuple(vocab.stoi.get(w, vocab.unk_id) for w in context + [target])
        groups["mapped_trigram:" + ("seen" if reference.ngram_counts[3][mapped] else "unseen")].append(index)
        groups["mapped_context:" + ("seen" if reference.context_totals[3][mapped[:-1]] else "unseen")].append(index)
    for name in ("target:eos", "target:in_vocabulary", "target:oov"):
        groups.setdefault(name, [])
    return groups


def select_cases(events, vocab, raw_counts, full_scores, quotas=CATEGORIES):
    scores = {model: [value for (m, _), value in full_scores.items() if m == model] for model in ("lstm", "ffnn")}
    if any(not values for values in scores.values()):
        raise ValueError("Both neural architectures are required for case selection")
    mean_logp = {model: np.mean([value[0] for value in values], axis=0) for model, values in scores.items()}
    all_wrong = {
        model: np.all([value[1] != np.array([vocab.stoi.get(e["target"], vocab.unk_id) for e in events])
                       for value in values], axis=0)
        for model, values in scores.items()
    }
    candidates = defaultdict(list)
    for event in events:
        if event["eos"]:
            continue
        i = event["event_index"]
        target = event["target"]
        gram = tuple(event["context"] + [target])
        count = raw_counts[3][gram]
        if count > 5:
            continue
        if target in vocab and count == 0 and all(word in vocab for word in event["context"]):
            candidates["known_word_unseen_combination"].append(i)
        if target not in vocab:
            candidates["oov_target"].append(i)
        if target in vocab and mean_logp["ffnn"][i] > mean_logp["lstm"][i]:
            candidates["ffnn_wins"].append(i)
        if target in vocab and all_wrong["lstm"][i] and all_wrong["ffnn"][i]:
            candidates["shared_failure"].append(i)
    selected = []
    used_sentences = set()
    availability = {}
    for category, quota in quotas:
        ordered = sorted(candidates[category], key=lambda i: fingerprint(["inaugural_cases_v1", events[i]["event_id"]]))
        available = [i for i in ordered if events[i]["sentence_index"] not in used_sentences]
        availability[category] = {"qualifying_events": len(ordered),
                                  "available_sentences_before_selection": len({events[i]["sentence_index"] for i in available}),
                                  "requested": quota}
        chosen = []
        for i in available:
            sentence = events[i]["sentence_index"]
            if sentence not in used_sentences:
                chosen.append((category, i))
                used_sentences.add(sentence)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError(f"Insufficient distinct sentences for {category}: {len(chosen)}/{quota}; no silent quota substitution")
        selected.extend(chosen)
    return selected, availability


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["budget_percent"], row["model"], row["group"])].append(row)
    result = []
    for (budget, model, group), members in sorted(groups.items()):
        if len({r["events"] for r in members}) != 1:
            raise ValueError("Seeds do not score identical group populations")
        row = dict(budget_percent=budget, model=model, group=group, seeds=len(members),
                   events=members[0]["events"], event_fraction=members[0]["event_fraction"],
                   status=members[0]["status"])
        for metric in ("nll", "perplexity", "accuracy", "nll_contribution", "zero_probability_events"):
            values = [r[metric] for r in members]
            row[metric + "_mean"] = None
            row[metric + "_std"] = None
            if all(value is not None for value in values):
                row[metric + "_mean"] = statistics.mean(values)
                row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(row)
    return result


def render_report(output, cases, summary, protocol, verification):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    budgets = protocol["budgets_percent"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, model in zip(axes, ("lstm", "ffnn")):
        base = np.zeros(len(budgets))
        for label in ("in_vocabulary", "oov", "eos"):
            values = [next(r["nll_contribution_mean"] for r in summary
                           if r["budget_percent"] == b and r["model"] == model and r["group"] == "target:" + label)
                      for b in budgets]
            axis.bar([str(b) for b in budgets], values, bottom=base, label=label)
            base += values
        axis.set(title=model.upper(), xlabel="Training documents (%)", ylabel="Contribution to overall mean NLL")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "loss_decomposition.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    for model in ("lstm", "ffnn"):
        for cohort, style in (("target_cohort:known_all", "-"), ("sequence_cohort:known_all", "--")):
            points = [next((r for r in summary if r["budget_percent"] == b and r["model"] == model and r["group"] == cohort), None)
                      for b in budgets]
            if all(p is not None for p in points):
                axis.errorbar(budgets, [p["nll_mean"] for p in points],
                              yerr=[p["nll_std"] or 0 for p in points], linestyle=style, marker="o",
                              label=model + " / " + cohort.split(":")[0])
    axis.set(xlabel="Training documents (%)", ylabel="Mean NLL on identical test occurrences",
             title="Stable known-word and known-sequence cohorts; seed SD")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "stable_cohorts.png", dpi=150)
    plt.close(figure)

    def table(rows, columns):
        heading = "".join("<th>" + html.escape(c) + "</th>" for c in columns)
        body = []
        for row in rows:
            values = [f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]) for c in columns]
            body.append("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in values) + "</tr>")
        return "<table><tr>" + heading + "</tr>" + "".join(body) + "</table>"

    sections = [
        "<h1>Inaugural: 100 stratified cases and all-budget loss analysis</h1>",
        "<p>Scoring only: frozen best-validation checkpoints, unchanged test data, no training or selection on new test scores. "
        "This is exploratory, outcome-stratified analysis of a previously inspected test set, not an unbiased error-rate sample. "
        "Original source casing is retained for display; counts and model inputs use lowercase/whitespace normalization.</p>",
        "<p>30 known-word unseen combinations, 30 OOV targets, 20 FFNN wins, 20 shared failures. "
        "All have original full-budget trigram count at most five; EOS is excluded. "
        "Selection uses SHA-256 ordering of stable source-event IDs, in that category order, "
        "without event or sentence reuse. Categories can qualify for multiple pools but selected cases are disjoint. "
        "FFNN wins means higher mean target log probability across all three seeds, not necessarily correct top-1. "
        "Shared failures require every seed of both architectures to predict incorrectly.</p>",
        "<p>Probabilities are P(mapped target | mapped context). Predicting UNK is not recovering an OOV word. "
        "FFNN has context two; LSTM sees the complete sentence prefix. BOS/EOS handling and prediction support match the frozen evaluation.</p>",
        "<h2>All-budget loss decomposition</h2>",
        "<p>For each seed, overall mean NLL equals the sum of target-category NLL contributions "
        "(category loss sum divided by all scored targets). Perplexity is exp(NLL); category perplexities must not be averaged. "
        "Seed summaries report arithmetic means and sample SDs; exp(mean seed NLL) differs from mean seed perplexity. "
        "MLE zeros remain infinite, represented by explicit status and blank numerical loss/PPL fields.</p>",
        '<img src="loss_decomposition.png" alt="Weighted NLL contributions">',
        table([r for r in summary if r["group"].startswith("target:")],
              ["budget_percent", "model", "group", "events", "event_fraction", "nll_mean", "nll_std", "nll_contribution_mean", "status"]),
        "<h2>Stable original-token event cohorts</h2>",
        "<p>Cohorts use the same source sentence and target position at every budget, not newly sampled words. "
        "known_all means retained in every vocabulary; oov_all means absent throughout; enters_at_X records first retention. "
        "Target cohorts use only target membership; sequence cohorts require both preceding words plus target "
        "(BOS is permitted). EOS is separate. Stable cohorts do not remove changing output support, "
        "LSTM's longer-prefix OOV exposure, parameter-count changes or training-budget confounding.</p>",
        '<img src="stable_cohorts.png" alt="Stable event cohorts">',
        table([r for r in summary if r["group"].startswith("target_cohort:") and r["model"] in ("lstm", "ffnn")],
              ["budget_percent", "model", "group", "events", "nll_mean", "nll_std", "accuracy_mean"]),
        "<h2>100 case studies</h2>",
    ]
    for case in cases:
        e = case["event"]
        sections.extend([
            f"<h3>{case['case_number']}. {html.escape(case['category'])}: {html.escape(e['original_target'])}</h3>",
            f"<p>Source: {html.escape(e['source'])}; original sentence position {e['sentence_position']}; "
            f"target position {e['position']} (both zero-based). Event: {html.escape(e['event_id'])}</p>",
            "<blockquote>" + html.escape(e["sentence"]) + "</blockquote>",
            "<p>" + html.escape(case["explanation"]) + "</p>",
            table(case["counts"], ["budget_percent", "normalized_target", "mapped_target", "target_count", "context_count",
                                   "bigram_count", "trigram_count", "mapped_target_count", "mapped_context_count", "mapped_trigram_count"]),
            table(case["scores"], ["budget_percent", "model", "seed", "mapped_target", "target_probability",
                                   "target_log_probability", "prediction", "correct"]),
            "<details><summary>Exact model-visible contexts by budget</summary><pre>" +
            html.escape(json.dumps(case["contexts"], indent=2)) + "</pre></details>",
        ])
    sections.extend([
        "<h2>Verification</h2><pre>" + html.escape(json.dumps(verification, indent=2)) + "</pre>",
        "<h2>Reproducible protocol</h2><pre>" + html.escape(json.dumps(protocol, indent=2)) + "</pre>",
        "<p>Machine-readable outputs: cases.json, case_scores.csv, case_counts.csv, decomposition.csv, "
        "decomposition_summary.csv, events.csv, scores.npz and verification.json. All seed results are retained. "
        "Outcome-selected examples cannot establish population superiority or statistical significance.</p>",
    ])
    style = "body{font:15px system-ui;max-width:1300px;margin:30px auto;padding:15px}table{border-collapse:collapse;display:block;overflow:auto;margin:15px 0}td,th{padding:5px 9px;border:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}blockquote{background:#f4f6f7;padding:15px}img{max-width:100%}h3{margin-top:50px}"
    (output / "report.html").write_text("<!doctype html><meta charset='utf-8'><title>Inaugural case analysis</title><style>" +
                                       style + "</style>" + "\n".join(sections), encoding="utf-8")


def analyze(experiment, output, device_name="auto", corpus="inaugural"):
    started = time.perf_counter()
    experiment = Path(experiment).resolve()
    output = Path(output).resolve()
    if output == experiment or output.is_relative_to(experiment) or experiment.is_relative_to(output):
        raise ValueError("Analysis output must be separate from the frozen experiment")
    output.mkdir(parents=True, exist_ok=False)
    config, prepared, seal, verified = verified_inputs(experiment, output, corpus)
    budgets = config["budgets_percent"]
    if budgets != [10, 50, 100]:
        raise ValueError("This declared analysis requires the three budgets [10,50,100]")
    specs = {spec["id"]: spec for spec in config["models"]}
    if set(specs) != set(MODELS):
        raise ValueError("Expected the five frozen PS-1 systems")
    for model in ("lstm", "ffnn"):
        if specs[model]["final_seeds"] != [42, 43, 44]:
            raise ValueError("Expected all three declared neural seeds")
    device = get_device(device_name)
    vocabs = {b: Vocab.from_list(read_json(experiment / corpus / f"budget_{b:03}" / "vocab.json")) for b in budgets}
    test = sentences_for(prepared, "test")
    test_words = list(test)
    events = make_events(test, prepared, vocabs, budgets)
    protocol = {
        "schema_version": 1, "name": "inaugural_analysis100_v1", "training_performed": False,
        "experiment_seal": seal["fingerprint"], "input_bundle_sha256": file_sha256(experiment / "input_bundle.zip"),
        "analysis_script_sha256": file_sha256(__file__), "prepared_fingerprint": prepared["fingerprint"],
        "budgets_percent": budgets, "neural_seeds": [42, 43, 44], "categories": dict(CATEGORIES),
        "case_reference_budget": 100, "rare_rule": "original normalized trigram count <=5; EOS excluded",
        "selection_order": [name for name, _ in CATEGORIES],
        "selection_sort": "SHA256 JSON fingerprint(['inaugural_cases_v1', stable source event ID]) ascending",
        "diversity": "one selected target per source sentence, globally across all categories",
        "win_metric": "strictly higher mean target log probability across all three seeds, no seed cherry-picking",
        "shared_failure": "all six neural top-1 predictions wrong on an in-vocabulary target",
        "cohorts": "fixed original test occurrences; target and preceding-two-word-plus-target vocabulary membership",
        "aggregation": "token-weighted NLL; arithmetic seed means and sample SD; UNK/EOS distinct",
        "exploratory": True, "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "verified_input_files": len(verified),
    }
    atomic_json(output / "protocol.json", protocol)
    decomposition = []
    scores = {}
    references = {}
    counts_by_budget = {}
    target_counts = {}
    case_context_counts = {}
    max_padded = config["resources"]["max_padded_tokens"]
    comparisons = []
    for b in budgets:
        vocab = vocabs[b]
        train = list(sentences_for(prepared, "train", b))
        counts_by_budget[b] = original_counts(train)
        target_counts[b] = Counter(w for sentence in train for w in sentence)
        case_context_counts[b] = Counter()
        for gram, count in counts_by_budget[b][3].items():
            case_context_counts[b][gram[:-1]] += count
        targets = np.array([vocab.stoi.get(e["target"], vocab.unk_id) for e in events])
        budget_directory = experiment / corpus / f"budget_{b:03}"
        with (budget_directory / "final/trigram_mle/seed0/counts.pkl").open("rb") as stream:
            references[b] = pickle.load(stream).model
        groups = group_indices(events, vocab, counts_by_budget[b], references[b])
        for model_id in MODELS:
            spec = specs[model_id]
            seeds = [42, 43, 44] if spec["family"] == "neural" else [0]
            for seed in seeds:
                directory = budget_directory / "final" / model_id / f"seed{seed}"
                result = read_json(directory / "result.json")
                if result["vocab_hash"] != vocab.fingerprint or result["prepared_fingerprint"] != prepared["fingerprint"]:
                    raise ValueError("Checkpoint, vocabulary and prepared-data identities disagree")
                saved = read_json(directory / "test_metrics.json")
                evaluation_status = read_json(directory / "evaluation_status.json")
                if file_sha256(directory / "test_metrics.json") != evaluation_status["sha256"]:
                    raise ValueError("Saved test metric checksum mismatch")
                if saved["seal_fingerprint"] != seal["fingerprint"]:
                    raise ValueError("Saved test evaluation is not bound to this seal")
                if spec["family"] == "neural":
                    model = build_model(spec.get("model", model_id), len(vocab), vocab.pad_id, **spec["architecture"])
                    model.load_state_dict(torch.load(directory / result["checkpoint"], map_location="cpu", weights_only=True))
                    model.to(device)
                    iterator = neural_events(model, vocab, test_words, device, max_padded_tokens=max_padded)
                else:
                    with (directory / result["checkpoint"]).open("rb") as stream:
                        model = pickle.load(stream)
                    iterator = ngram_events(model, vocab, test_words)
                logp = np.empty(len(events), dtype=np.float64)
                predictions = np.empty(len(events), dtype=np.int32)
                scored = 0
                for scored, event in enumerate(iterator, start=1):
                    expected = events[scored - 1]
                    if (event["sentence_index"], event["position"], event["target"]) != (
                        expected["sentence_index"], expected["position"], targets[scored - 1]
                    ):
                        raise ValueError("Scored stream is not aligned with original source events")
                    logp[scored - 1] = event["log_probability"]
                    predictions[scored - 1] = -1 if event["prediction"] is None else event["prediction"]
                if scored != len(events):
                    raise ValueError("Incomplete scored target stream")
                scores[(b, model_id, seed)] = (logp, predictions)
                for group, indices in groups.items():
                    metrics = summarize_losses(logp, predictions, targets, indices, len(events))
                    decomposition.append(dict(budget_percent=b, model=model_id, seed=seed, group=group, **metrics))
                overall = next(r for r in reversed(decomposition) if r["group"] == "overall")
                if overall["events"] != saved["tokens"] or overall["zero_probability_events"] != saved["zero_probability_events"]:
                    raise ValueError("Scoring target counts or MLE zeros differ from archived evaluation")
                if saved["nll"] is not None and not math.isclose(overall["nll"], saved["nll"], rel_tol=2e-6, abs_tol=2e-6):
                    raise ValueError(f"Rescored NLL differs from archived evaluation: {b}/{model_id}/{seed}")
                parts = [r for r in decomposition[-len(groups):] if r["group"].startswith("target:")]
                if sum(r["events"] for r in parts) != len(events) or not math.isclose(
                    sum(r["finite_nll_sum"] for r in parts), overall["finite_nll_sum"], rel_tol=1e-12, abs_tol=1e-8
                ):
                    raise ValueError("Target-category NLL decomposition does not reconstruct the total")
                comparisons.append(dict(budget_percent=b, model=model_id, seed=seed,
                                        archived_nll=saved["nll"], rescored_nll=overall["nll"],
                                        events=scored, zero_probability_events=overall["zero_probability_events"]))
                print(f"Scored {b}% {model_id} seed{seed}: {scored} events, NLL {overall['nll']}", flush=True)
                del model
    full_scores = {(m, seed): value for (b, m, seed), value in scores.items() if b == 100 and m in ("lstm", "ffnn")}
    selected, availability = select_cases(events, vocabs[100], counts_by_budget[100], full_scores)
    cases = []
    flat_scores = []
    flat_counts = []
    for number, (category, i) in enumerate(selected, start=1):
        e = events[i]
        case = dict(case_number=number, category=category, event={k: v for k, v in e.items() if k != "words"},
                    counts=[], contexts=[], scores=[])
        for b in budgets:
            vocab = vocabs[b]
            prefix = e["words"][:e["position"]]
            mapped_prefix = [vocab.itos[vocab.stoi.get(w, vocab.unk_id)] for w in prefix]
            mapped_target = vocab.itos[vocab.stoi.get(e["target"], vocab.unk_id)]
            mapped_gram = tuple(vocab.stoi.get(w, vocab.unk_id) for w in e["context"] + [e["target"]])
            counts = dict(budget_percent=b, normalized_target=e["target"], mapped_target=mapped_target,
                          target_count=target_counts[b][e["target"]], context_count=case_context_counts[b][tuple(e["context"])],
                          bigram_count=counts_by_budget[b][2][tuple(e["context"][-1:] + [e["target"]])],
                          trigram_count=counts_by_budget[b][3][tuple(e["context"] + [e["target"]])],
                          mapped_target_count=references[b].ngram_counts[1][(mapped_gram[-1],)],
                          mapped_context_count=references[b].context_totals[3][mapped_gram[:-1]],
                          mapped_trigram_count=references[b].ngram_counts[3][mapped_gram])
            case["counts"].append(counts)
            flat_counts.append(dict(case_number=number, category=category, event_id=e["event_id"], **counts))
            case["contexts"].append(dict(budget_percent=b, lstm_prefix=[BOS] + mapped_prefix,
                                         ffnn_context=([BOS, BOS] + mapped_prefix)[-2:]))
            for model_id in MODELS:
                for seed in ([42, 43, 44] if model_id in ("lstm", "ffnn") else [0]):
                    logp, prediction_ids = scores[(b, model_id, seed)]
                    prediction = None if prediction_ids[i] == -1 else vocab.itos[prediction_ids[i]]
                    score = dict(budget_percent=b, model=model_id, seed=seed, mapped_target=mapped_target,
                                 target_probability=math.exp(logp[i]), target_log_probability=None if np.isneginf(logp[i]) else float(logp[i]),
                                 probability_status="zero_probability" if np.isneginf(logp[i]) else "finite",
                                 prediction=prediction, correct=prediction == mapped_target)
                    case["scores"].append(score)
                    flat_scores.append(dict(case_number=number, category=category, event_id=e["event_id"], **score))
        if category == "known_word_unseen_combination":
            explanation = "All two-word-context tokens and the target are retained at full budget, but this original trigram never occurs in training. This tests an unseen combination rather than lexical OOV; it is not guaranteed to be predicted correctly."
        elif category == "oov_target":
            explanation = "The original target is outside the full-budget vocabulary. Its displayed probability and correctness concern pooled UNK, not recovery of that original word; mapped counts may combine many different lexical events."
        elif category == "ffnn_wins":
            explanation = "FFNN assigns higher mean log probability to this full-budget known-word target across all three seeds. This is a local probability advantage, not necessarily correct top-1 prediction or a reversal of the aggregate ranking."
        else:
            explanation = "Every seed of both neural models predicts the wrong full-budget known-word target. This illustrates a shared lexical failure; probability differences can still exist and are retained for every seed."
        case["explanation"] = explanation
        cases.append(case)
    summary = aggregate(decomposition)
    atomic_json(output / "cases.json", cases)
    write_csv(output / "case_scores.csv", flat_scores)
    write_csv(output / "case_counts.csv", flat_counts)
    write_csv(output / "decomposition.csv", decomposition)
    write_csv(output / "decomposition_summary.csv", summary)
    write_csv(output / "events.csv", [{k: e[k] for k in (
        "event_index", "event_id", "source", "sentence_index", "sentence_position", "position",
        "eos", "original_target", "target", "target_cohort", "sequence_cohort",
    )} for e in events])
    np.savez_compressed(output / "scores.npz", **{
        f"b{b}_{m}_s{seed}_{kind}": value
        for (b, m, seed), arrays in scores.items() for kind, value in zip(("logp", "prediction_id"), arrays)
    })
    verification = {
        "case_count": len(cases), "distinct_case_sentences": len({e["event"]["sentence_index"] for e in cases}),
        "case_categories": dict(Counter(c["category"] for c in cases)), "candidate_availability": availability,
        "test_sentences": len(test), "word_events": sum(not e["eos"] for e in events), "eos_events": len(test),
        "scored_events_per_run": len(events), "scored_runs": len(comparisons),
        "source_event_ids_unique": len({e["event_id"] for e in events}) == len(events),
        "all_target_partitions_reconstruct_nll": True, "all_rescored_nll_match_archive": True,
        "archived_comparisons": comparisons, "analysis_wall_seconds": time.perf_counter() - started,
    }
    for name, digest in verified.items():
        if file_sha256(experiment / name) != digest:
            raise ValueError("Frozen experiment changed during analysis")
    verification["frozen_training_artifacts_unchanged"] = True
    atomic_json(output / "verification.json", verification)
    render_report(output, cases, summary, protocol, verification)
    artifacts = {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    atomic_json(output / "artifact_manifest.json", {"protocol_fingerprint": fingerprint(protocol), "files": artifacts})
    print(f"ANALYSIS_COMPLETE: {output / 'report.html'}", flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    analyze(args.experiment, args.output, args.device)


if __name__ == "__main__":
    main()
