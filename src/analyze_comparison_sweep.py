"""Paper tables, figures and rare/unseen case studies from a finished comparison sweep.

Read-only: uses metrics.csv, perplexity_by_percentage.csv, runs.csv, data_stats.json,
model histories and (with --events) scoring_events.jsonl.gz. No model is fitted.

    python src/analyze_comparison_sweep.py                      # tables + figures
    python src/analyze_comparison_sweep.py --skip-summary --events --corpora inaugural
"""

import argparse
import gzip
import hashlib
import json
import math
import pickle
from array import array
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "results/comparison_sweep_20261102_v3/full/runs"
DEFAULT_OUT = ROOT / "results/paper_v3"
SUBMISSION_OUT = ROOT / "results/paper_submission_20261107"
RUN_DIRS = {"brown": "brown_press_neural", "gutenberg": "gutenberg_neural", "inaugural": "inaugural_neural",
            "reuters": "reuters_neural", "roc-stories": "roc-stories", "webtext": "webtext-nn"}
LABELS = {"brown": "Brown", "gutenberg": "Gutenberg", "inaugural": "Inaugural",
          "reuters": "Reuters", "roc-stories": "ROCStories", "webtext": "WebText"}
BUDGETS = [20, 40, 60, 80, 100]
MODELS = ["bigram_mle", "trigram_mle", "bigram_laplace", "trigram_laplace",
          "bigram_backoff", "trigram_backoff", "ffnn_ctx1", "ffnn_ctx2", "lstm"]
NEURAL = ["ffnn_ctx1", "ffnn_ctx2", "lstm"]
BACKOFF = ["bigram_backoff", "trigram_backoff"]
CURVE_MODELS = BACKOFF + NEURAL
SMOOTHING = {"mle": "none (MLE)", "laplace": "add-one", "backoff": "abs. discount backoff"}
CATEGORIES = ["seen_trigram", "unseen_trigram", "unseen_bigram", "oov_target"]
CATEGORY_LABELS = {"seen_trigram": "seen trigram", "unseen_trigram": "unseen trigram, seen bigram",
                   "unseen_bigram": "unseen bigram, known target", "oov_target": "OOV target"}
ZERO, OOV = 1, 2
CORRECT, TARGET_UNK, PREDICTED_UNK, HISTORY_SEEN, HISTORY_AVAILABLE = 4, 8, 16, 32, 64


def accuracy_flags(event):
    if event["correct"] != (event["target_id"] == event["prediction_id"]):
        raise ValueError("Saved correctness disagrees with target/prediction IDs")
    if event["eos"] and event["oov"]:
        raise ValueError("An EOS event cannot be an OOV word target")
    covered = event["context_covered"]
    if covered is not None and not isinstance(covered, bool):
        raise ValueError("Invalid saved context coverage")
    flags = CORRECT if event["correct"] else 0
    if event["mapped_target"] == "<unk>":
        flags |= TARGET_UNK
    if event["prediction_id"] == 1:
        flags |= PREDICTED_UNK
    if covered is not None:
        flags |= HISTORY_AVAILABLE
    if covered is True:
        flags |= HISTORY_SEEN
    return flags


def fmt(value, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:,.{digits}f}"


def fmt_pm(mean, std, digits=1):
    text = fmt(mean, digits)
    if text in ("inf", "-") or std is None or math.isnan(std):
        return text
    return f"{text} ± {std:,.{digits}f}"


def md_table(frame):
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "|" + "|".join("---" for _ in frame.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def write_md(path, title, sections):
    body = [f"# {title}", ""]
    for heading, frame, note in sections:
        body += [f"## {heading}", ""]
        if note:
            body += [note, ""]
        body += [md_table(frame), ""]
    path.write_text("\n".join(body), encoding="utf-8")


def mean_std(values):
    values = [float(v) for v in values]
    if any(math.isinf(v) for v in values):
        return math.inf, math.nan
    if len(values) < 2:
        return values[0], math.nan
    return float(np.mean(values)), float(np.std(values, ddof=1))


def submission_aggregate(conditions, fields):
    rows = []
    for (corpus, budget, model), group in conditions.groupby(["corpus", "budget_percent", "model"]):
        row = dict(corpus=corpus, budget_percent=budget, model=model,
                   conditions=len(group), recorded_seeds=",".join(sorted(group["seed"])))
        for field in fields:
            row[f"{field}_mean"], row[f"{field}_sd"] = mean_std(group[field])
        rows.append(row)
    return pd.DataFrame(rows)


def run_submission_figures(runs, out, corpora, accuracy_input):
    out = out.resolve()
    if runs.resolve() != DEFAULT_RUNS.resolve() or out.resolve() != SUBMISSION_OUT.resolve():
        raise ValueError("Final figures require canonical V3 runs and results/paper_submission_20261107")
    if len(corpora) != 6 or set(corpora) != set(RUN_DIRS):
        raise ValueError("Final figures require all six corpora exactly once")
    corpora = list(RUN_DIRS)
    inputs, validations, missing = {}, {}, []
    paper = ROOT / "results/paper_v3_six_corpora_20261106"
    laplace = ROOT / "results/paper_laplace_audit_20261107/tables"

    def load(path):
        if path.suffix not in (".csv", ".json") or path.stat().st_size > 2_000_000:
            raise ValueError(f"Not an eligible small CSV/JSON: {path}")
        content = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        if path.suffix == ".json":
            return json.loads(content)
        from io import BytesIO
        frame = pd.read_csv(BytesIO(content), dtype={"seed": str})
        if "seed" in frame:
            frame["seed"] = frame["seed"].fillna("").str.replace(r"\.0$", "", regex=True)
        return frame.rename(columns={"percent": "budget_percent"})

    def check(name, condition):
        if not bool(condition):
            raise ValueError(f"Submission reconciliation failed: {name}")
        validations[name] = True

    def close(name, actual, expected, tolerance=1e-10):
        check(name, np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                                rtol=tolerance, atol=tolerance, equal_nan=True))

    keys = ["corpus", "budget_percent", "model", "seed"]
    metric_frames, budget_rows, convergence_frames = [], [], []
    for corpus in corpora:
        directory = runs / RUN_DIRS[corpus]
        metrics = load(directory / "metrics/metrics.csv")
        check(f"{corpus}:metric_corpus", set(metrics["corpus"]) == {corpus})
        metrics["budget_percent"] = metrics["budget_percent"].astype(int)
        metric_frames.append(metrics)
        for budget in BUDGETS:
            stats = load(directory / f"train_{budget}pct/data_stats.json")
            check(f"{corpus}:{budget}:recorded_budget", stats["train_percent"] == budget)
            budget_rows.append(dict(corpus=corpus, budget_percent=budget, **stats))
            convergence = load(directory / f"train_{budget}pct/runs.csv")
            convergence["corpus"], convergence["budget_percent"] = corpus, budget
            convergence = convergence.rename(columns={"model": "neural_family", "variant": "model",
                                                      "test_seconds": "original_run_test_seconds",
                                                      "test_ppl": "original_run_test_perplexity"})
            convergence_frames.append(convergence)
    metrics = pd.concat(metric_frames, ignore_index=True)
    budgets = pd.DataFrame(budget_rows).drop(columns="train_percent")
    convergence = pd.concat(convergence_frames, ignore_index=True)
    test = metrics[metrics["source"] == "test"].copy()
    valid = metrics[metrics["source"] == "valid"].copy()
    expected = {(corpus, budget, model, seed)
                for corpus in corpora for budget in BUDGETS for model in MODELS
                for seed in (["13", "42", "2026"] if corpus in corpora[:4] else ["13"])
                if model in NEURAL}
    expected.update((corpus, budget, model, "") for corpus in corpora for budget in BUDGETS
                    for model in MODELS if model not in NEURAL)
    for name, frame in (("test", test), ("validation", valid)):
        check(f"{name}:390_unique_keys", len(frame) == 390 and set(map(tuple, frame[keys].values)) == expected)
        check(f"{name}:status_values", set(frame["status"]) == {"finite", "infinite_zero_probability"})
        frame.loc[frame["status"] == "infinite_zero_probability", "perplexity"] = math.inf
        check(f"{name}:finite_status", np.isfinite(frame.loc[frame["status"] == "finite", "perplexity"]).all())
        check(f"{name}:infinite_only_mle", set(frame.loc[frame["status"] != "finite", "model"])
              == {"bigram_mle", "trigram_mle"})
        check(f"{name}:60_infinite", (frame["status"] != "finite").sum() == 60)
        check(f"{name}:zero_probability_status", ((frame["zero_probability_events"] > 0)
              == (frame["status"] != "finite")).all())
    check("convergence:210_unique_keys", len(convergence) == 210 and
          set(map(tuple, convergence[keys].values)) == {key for key in expected if key[2] in NEURAL})
    accuracy = load(accuracy_input / "tables/accuracy_by_condition.csv")
    check("accuracy:390_unique_keys", len(accuracy) == 390 and set(map(tuple, accuracy[keys].values)) == expected)
    test = test.merge(accuracy, on=keys, validate="one_to_one")
    close("accuracy:original_overall", test["accuracy"], test["overall_accuracy"])
    close("accuracy:original_correct", test["correct_predictions"], test["overall_correct"], 0)
    close("accuracy:words_exclude_eos", test["word_tokens"], test["overall_words"], 0)
    close("accuracy:ppl_events_include_eos", test["prediction_events"], test["overall_words"] + test["eos_words"], 0)
    close("accuracy:known_plus_oov", test["overall_words"], test["in_vocabulary_words"] + test["oov_words"], 0)
    close("accuracy:oov_words", test["oov_tokens"], test["oov_words"], 0)
    close("accuracy:oov_rate_words_only", test["oov_rate"], test["oov_words"] / test["overall_words"])
    for population in ("overall", "in_vocabulary", "oov", "unk_targets", "excluding_unk_targets", "eos"):
        close(f"accuracy:{population}:ratio", test[f"{population}_accuracy"],
              test[f"{population}_correct"] / test[f"{population}_words"].replace(0, np.nan))
    for corpus in corpora:
        subset = budgets[budgets["corpus"] == corpus].sort_values("budget_percent")
        full_sentences = int(subset.iloc[-1]["train_sentences"])
        close(f"{corpus}:sentence_prefix_ceiling", subset["train_sentences"],
              [math.ceil(full_sentences * budget / 100) for budget in BUDGETS], 0)
        check(f"{corpus}:vocabulary_nondecreasing", (subset["vocab_size"].diff().dropna() >= 0).all())
        check(f"{corpus}:oov_nonincreasing", (subset["test_oov_rate"].diff().dropna() <= 0).all())
    for row in budgets.itertuples():
        selected = metrics[(metrics["corpus"] == row.corpus) & (metrics["budget_percent"] == row.budget_percent)]
        for split, tokens, sentences in (("train", row.train_tokens, row.train_sentences),
                                         ("valid", row.valid_tokens, row.valid_sentences),
                                         ("test", row.test_tokens, row.test_sentences)):
            split_rows = selected[selected["source"] == split]
            close(f"{row.corpus}:{row.budget_percent}:{split}:words", split_rows["word_tokens"], tokens, 0)
            close(f"{row.corpus}:{row.budget_percent}:{split}:sentences", split_rows["sentences"], sentences, 0)
        neural_rows = convergence[(convergence["corpus"] == row.corpus)
                                  & (convergence["budget_percent"] == row.budget_percent)]
        close(f"{row.corpus}:{row.budget_percent}:neural_vocab", neural_rows["vocab_size"], row.vocab_size, 0)
        close(f"{row.corpus}:{row.budget_percent}:neural_test_events", neural_rows["test_tokens"],
              row.test_tokens + row.test_sentences, 0)
        close(f"{row.corpus}:{row.budget_percent}:test_oov", selected[selected["source"] == "test"]["oov_rate"],
              row.test_oov_rate)
    neural_join = test[test["model"].isin(NEURAL)].merge(convergence, on=keys, validate="one_to_one",
                                                      suffixes=("", "_run"))
    close("convergence:fit_times", neural_join["train_seconds"], neural_join["train_seconds_run"])
    original_ppl_relative_error = float(np.max(np.abs(neural_join["original_run_test_perplexity"]
                                                     / neural_join["perplexity"] - 1)))
    check("convergence:original_test_ppl_within_1e-4_relative", original_ppl_relative_error < 1e-4)
    test = test.merge(valid[keys + ["perplexity", "status", "evaluation_seconds"]].rename(columns={
        "perplexity": "validation_perplexity", "status": "validation_perplexity_status",
        "evaluation_seconds": "comparison_validation_seconds"}), on=keys, validate="one_to_one")
    test = test.rename(columns={"word_tokens": "test_word_tokens", "sentences": "test_sentences",
        "prediction_events": "test_prediction_events_including_eos", "correct_predictions": "test_correct_words",
        "accuracy": "test_overall_accuracy", "perplexity": "test_perplexity", "status": "test_perplexity_status",
        "zero_probability_events": "test_zero_probability_events", "oov_tokens": "test_oov_word_tokens",
        "oov_rate": "test_oov_word_rate", "evaluation_seconds": "comparison_test_seconds",
        "train_seconds": "fit_train_seconds", "source": "evaluation_split", "eos_words": "eos_events"})
    test["fit_cost_scope"] = np.where(test["model"].isin(NEURAL), "one neural condition",
                                     "one shared counting pass per corpus/budget; repeated across six variants")
    test["test_zero_probability_fraction_including_eos"] = (test["test_zero_probability_events"]
                                                            / test["test_prediction_events_including_eos"])
    fields = [field for field in test.select_dtypes(include="number").columns
              if field not in ("budget_percent", "order")]
    summary = submission_aggregate(test, fields)
    check("summary:270_unique_keys", len(summary) == 270 and not summary.duplicated(keys[:3]).any())
    published = load(paper / "tables/test_metrics_by_budget.csv").set_index(keys[:3]).sort_index()
    indexed = summary.set_index(keys[:3]).sort_index()
    check("summary:published_keys", indexed.index.equals(published.index))
    for current, original in (("test_perplexity_mean", "ppl_mean"), ("test_perplexity_sd", "ppl_std"),
                              ("test_overall_accuracy_mean", "accuracy_mean"),
                              ("test_overall_accuracy_sd", "accuracy_std"),
                              ("fit_train_seconds_mean", "train_seconds_mean"),
                              ("fit_train_seconds_sd", "train_seconds_std"),
                              ("comparison_test_seconds_mean", "evaluation_seconds_mean"),
                              ("validation_perplexity_mean", "valid_ppl_mean")):
        close(f"summary:published:{original}", indexed[current], published[original])
    published_accuracy = load(accuracy_input / "tables/accuracy_summary.csv").set_index(keys[:3]).sort_index()
    check("summary:audited_accuracy_keys", indexed.index.equals(published_accuracy.index))
    for field in accuracy.select_dtypes(include="number").columns:
        if f"{field}_mean" in published_accuracy:
            renamed = field.replace("eos_words", "eos_events")
            close(f"summary:accuracy:{field}:mean", indexed[f"{renamed}_mean"], published_accuracy[f"{field}_mean"])
            close(f"summary:accuracy:{field}:sd", indexed[f"{renamed}_sd"], published_accuracy[f"{field}_std"])
    baseline = load(laplace / "always_unk_baseline.csv").rename(columns={
        "accuracy": "always_unk_accuracy", "words": "test_words", "correct": "mapped_unk_target_words"})
    check("baseline:30_unique_keys", len(baseline) == 30 and not baseline.duplicated(keys[:2]).any())
    baseline_join = test.merge(baseline, on=keys[:2], validate="many_to_one")
    close("baseline:words", baseline_join["overall_words"], baseline_join["test_words"], 0)
    close("baseline:targets", baseline_join["unk_targets_words"], baseline_join["mapped_unk_target_words"], 0)
    close("baseline:accuracy", baseline["always_unk_accuracy"], baseline["mapped_unk_target_words"] / baseline["test_words"])
    vocabulary_join = budgets.merge(baseline, on=keys[:2], validate="one_to_one")
    close("baseline:allowed_vocabulary", vocabulary_join["vocab_size"], vocabulary_join["allowed_targets"] + 2, 0)
    budgets["budget_unit"] = "ceil sentence prefix; selected before training vocabulary construction"
    budgets["vocab_min_frequency"] = np.where(budgets["corpus"] == "brown", 2, 1)
    budgets = budgets.rename(columns={"train_tokens": "selected_train_word_tokens",
        "train_sentences": "selected_train_sentences", "valid_tokens": "validation_word_tokens",
        "valid_sentences": "validation_sentences", "test_tokens": "test_word_tokens",
        "vocab_size": "vocabulary_size_including_specials", "valid_oov_rate": "validation_oov_word_rate",
        "test_oov_rate": "test_oov_word_rate"})
    timing_rows = []
    for (corpus, budget), group in test.groupby(keys[:2]):
        statistical = group[~group["model"].isin(NEURAL)]
        check(f"{corpus}:{budget}:shared_count_fit", len(statistical) == 6
              and statistical["fit_train_seconds"].nunique(dropna=False) == 1)
        timing_rows.append(dict(corpus=corpus, budget_percent=budget, shared_count_fit_passes=1,
            shared_count_fit_seconds=statistical["fit_train_seconds"].iloc[0],
            neural_conditions=len(group) - 6,
            neural_fit_seconds_sum=group[group["model"].isin(NEURAL)]["fit_train_seconds"].sum(min_count=1),
            statistical_comparison_test_seconds_sum=statistical["comparison_test_seconds"].sum(min_count=1),
            neural_comparison_test_seconds_sum=group[group["model"].isin(NEURAL)]["comparison_test_seconds"].sum(min_count=1)))
    timing = pd.DataFrame(timing_rows)
    allocation_path = ROOT / ("results/comparison_sweep_20261102_v3/comparison_attempts/"
                              "kubeflow_recovery_20261103/nlp_environment.json")
    allocation = load(allocation_path)
    provenance = pd.DataFrame([dict(corpus=corpus,
        fit_field="metrics.csv:train_seconds; neural reconciled to runs.csv:train_seconds",
        comparison_test_field="metrics.csv:evaluation_seconds where source=test",
        original_test_field="runs.csv:test_seconds; distinct earlier evaluation, not plotted or added",
        observed_recovery_gpu=allocation.get("gpu", "") if corpus in ("gutenberg", "inaugural") else "",
        observed_recovery_torch=allocation.get("torch", "") if corpus in ("gutenberg", "inaugural") else "",
        observation_scope="recovery session allocation, not a per-condition hardware log"
                          if corpus in ("gutenberg", "inaugural") else "no hardware attribution in loaded condition CSVs",
        per_condition_training_device="", per_condition_comparison_device="", hardware_normalized=False)
        for corpus in corpora])
    missing.append("Per-condition CPU/GPU model, device, library/runtime and utilization are not recorded in the loaded CSVs; recovery allocation is not a timing-normalization claim.")
    categories = []
    for corpus in corpora:
        frame = load(paper / f"tables/event_categories_{corpus}.csv")
        check(f"{corpus}:20_category_rows", len(frame) == 20 and not frame.duplicated(["budget_percent", "category"]).any())
        check(f"{corpus}:category_names", set(frame["category"]) == set(CATEGORY_LABELS.values()))
        totals = frame.groupby("budget_percent")["events"].sum()
        event_counts = test[test["corpus"] == corpus].groupby("budget_percent")["test_prediction_events_including_eos"].first()
        close(f"{corpus}:category_event_totals", totals, event_counts, 0)
        close(f"{corpus}:category_shares", frame["share"], frame["events"] / frame["budget_percent"].map(totals))
        for row in frame.itertuples():
            for model in MODELS:
                categories.append(dict(corpus=corpus, budget_percent=row.budget_percent, category=row.category,
                    events_including_eos=row.events, event_share_including_eos=row.share, model=model,
                    conditional_perplexity_mean=getattr(row, f"{model}_ppl"),
                    conditional_perplexity_sd=getattr(row, f"{model}_std")))
    categories = pd.DataFrame(categories)
    loss = load(paper / "tables/full_budget_loss_decomposition.csv")
    chosen = loss[loss["is_full_budget_best_backoff"] & loss["category"].isin(CATEGORIES)].copy()
    check("loss:full_budget", (chosen["budget_percent"] == 100).all())
    check("loss:56_unique_best_backoff_seed_categories", len(chosen) == 56
          and not chosen.duplicated(["corpus", "seed", "category"]).any())
    for (corpus, seed), group in chosen.groupby(["corpus", "seed"]):
        comparator = str(group.iloc[0]["backoff"])
        full = test[(test["corpus"] == corpus) & (test["budget_percent"] == 100)]
        best = summary[(summary["corpus"] == corpus) & (summary["budget_percent"] == 100)
                       & summary["model"].isin(BACKOFF)].sort_values("test_perplexity_mean").iloc[0]["model"]
        check(f"{corpus}:{seed}:full_best_backoff", comparator == best and group["backoff"].nunique() == 1)
        close(f"{corpus}:{seed}:loss_shares", group["share"].sum(), 1)
        close(f"{corpus}:{seed}:loss_weights", group["weighted_nll_delta"],
              group["share"] * (group["lstm_nll"] - group["backoff_nll"]))
        lstm_ppl = float(full[(full["model"] == "lstm") & (full["seed"] == seed)].iloc[0]["test_perplexity"])
        backoff_ppl = float(full[full["model"] == comparator].iloc[0]["test_perplexity"])
        close(f"{corpus}:{seed}:loss_total_to_metric_nll", group["weighted_nll_delta"].sum(),
              math.log(lstm_ppl / backoff_ppl), 1e-6)
    loss_summary_rows = []
    for (corpus, category), group in chosen.groupby(["corpus", "category"]):
        mean, sd = mean_std(group["weighted_nll_delta"])
        loss_summary_rows.append(dict(corpus=corpus, category=category, backoff=group.iloc[0]["backoff"],
            seeds=len(group), events_including_eos=int(group.iloc[0]["events"]),
            event_share_including_eos=float(group.iloc[0]["share"]),
            weighted_nll_delta_mean=mean, weighted_nll_delta_sd=sd))
    loss_summary = pd.DataFrame(loss_summary_rows)
    contexts = load(accuracy_input / "tables/context_accuracy_summary.csv")
    ties = load(laplace / "laplace_tie_attribution.csv")
    check("context:4860_unique_rows", len(contexts) == 4860 and not contexts.duplicated(
        keys[:3] + ["history_basis", "history_state", "target_group"]).any())
    check("laplace:60_unique_rows", len(ties) == 60 and not ties.duplicated(keys[:3]).any())
    frames = {"test_metrics_by_condition.csv": test, "test_metrics_seed_summary.csv": summary,
              "budget_data_sizes.csv": budgets, "neural_convergence_by_condition.csv": convergence,
              "recorded_cost_by_budget.csv": timing, "runtime_provenance.csv": provenance,
              "always_unk_baseline.csv": baseline, "context_accuracy_seed_summary.csv": contexts,
              "laplace_tie_attribution.csv": ties, "event_category_metrics.csv": categories,
              "full_budget_loss_by_seed.csv": loss, "full_budget_loss_seed_summary.csv": loss_summary}
    for folder in ("tables", "figures"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out / "tables" / name, index=False)
    figures = plot_submission_figures(out, corpora, summary, budgets, baseline, timing, categories, loss_summary)
    reused_path = paper / "figures/neural_vs_backoff_ratio.png"
    reused = dict(path=str(reused_path.relative_to(ROOT)), bytes=reused_path.stat().st_size,
                  sha256=hashlib.sha256(reused_path.read_bytes()).hexdigest(), copied=False,
                  caption="Best neural / best backoff PPL across all six corpora; the winning neural variant may change with budget.")
    definitions = []
    for name, frame in frames.items():
        for field in frame.columns:
            unit = "recorded value"
            if "seconds" in field:
                unit = "wall seconds; mixed recorded runtimes, not hardware-normalized"
            elif "accuracy" in field or "fraction" in field or "share" in field or "oov" in field and "rate" in field:
                unit = "fraction, not percentage; blank when denominator is zero"
            elif "perplexity" in field:
                unit = "PPL on words + EOS; inf is preserved; category PPL is conditional"
            elif "nll" in field:
                unit = "natural-log loss; weighted delta = share * (LSTM NLL - fixed backoff NLL)"
            elif "words" in field or "word_tokens" in field:
                unit = "word targets only, excludes EOS"
            elif "events" in field or field.startswith("eos_"):
                unit = "prediction events; includes EOS where stated, eos_* is EOS only"
            elif field == "seed":
                unit = "13/42/2026 neural; blank statistical, not seed zero"
            if field.endswith("_sd") or field.endswith("_std"):
                unit += "; sample SD (ddof=1); blank for one run or infinite PPL"
            definitions.append(dict(table=name, field=field, definition=unit))
    definition_frame = pd.DataFrame(definitions)
    definition_frame.to_csv(out / "tables/field_definitions.csv", index=False)
    frames["field_definitions.csv"] = definition_frame
    outputs = {}
    for name, frame in frames.items():
        path = out / "tables" / name
        outputs[str(path.relative_to(out))] = dict(rows=len(frame), fields=list(frame.columns),
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for figure in figures:
        for suffix in ("png", "pdf"):
            path = out / "figures" / f"{figure['name']}.{suffix}"
            outputs[str(path.relative_to(out))] = dict(bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    png_bytes = sum(record["bytes"] for name, record in outputs.items() if name.endswith(".png"))
    check("figures:total_png_under_2MB", png_bytes <= 2_000_000)
    source_path = Path(__file__)
    inputs[str(source_path.relative_to(ROOT))] = dict(bytes=source_path.stat().st_size,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), role="generator source, not an experimental input")
    report = dict(step=9, status="passed", input_scope="named small CSV/JSON only; no scoring archives",
        canonical_runs=str(runs.relative_to(ROOT)), output=str(out.relative_to(ROOT)), inputs=inputs, outputs=outputs,
        validation_checks=validations, original_vs_comparison_neural_ppl_max_relative_error=original_ppl_relative_error,
        original_vs_comparison_neural_ppl_relative_tolerance=1e-4,
        png_bytes_total=png_bytes, figures=figures, reused_figures=[reused], missing_evidence=missing,
        experimental_contract=dict(budgets=BUDGETS, budget_unit="ceil sentence prefixes",
            controlling_route="run_full_config -> run_from_config -> selected corpus.train -> LanguageModelController -> Vocab(corpus.train)",
            vocabulary="per budget; Brown min_freq=2, other corpora=1; PAD/BOS excluded from target support",
            ppl_population="words + EOS", accuracy_and_oov_population="words only",
            neural_seeds={corpus: [13, 42, 2026] if corpus in corpora[:4] else [13] for corpus in corpora},
            statistical_seed="", infinite_test_conditions=60, seed_sd="sample ddof=1; no SD for single runs",
            empty_subgroup_accuracy="undefined, exported blank", shared_count_cost="one pass per corpus/budget, never sum six copies"),
        no_fitting_scoring_performed=True, operations=dict(training=False, fitting=False, checkpoint_loading=False,
            model_building=False, neural_forward=False, neural_inference=False, probability_rescoring=False,
            scoring_archive_scans=False, installs=False, downloads=False),
        renderer=dict(matplotlib=matplotlib.__version__, numpy=np.__version__, pandas=pd.__version__,
            runtime_scope="current local figure generation only, not original experiment environment"))
    (out / "graph_checks.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(out), test_conditions=len(test), seed_aggregates=len(summary),
                          convergence_conditions=len(convergence), png_bytes=png_bytes, checks=len(validations))), flush=True)


def plot_submission_figures(out, corpora, summary, budgets, baseline, timing, categories, loss):
    names = {"bigram_mle": "B-MLE", "trigram_mle": "T-MLE", "bigram_laplace": "B-Laplace",
             "trigram_laplace": "T-Laplace", "bigram_backoff": "B-backoff", "trigram_backoff": "T-backoff",
             "ffnn_ctx1": "FFNN-1", "ffnn_ctx2": "FFNN-2", "lstm": "LSTM"}
    colors = dict(zip(MODELS, ["#777777", "#333333", "#c49a00", "#c66c18", "#137c64", "#38a6a5",
                               "#3269a8", "#b95783", "#a43129"]))
    category_colors = ["#137c64", "#38a6a5", "#3269a8", "#c66c18"]
    figures = []

    def panels(height=5.4, right=.94, wspace=.57, left=.11):
        figure, axes = plt.subplots(2, 3, figsize=(7.2, height))
        figure.subplots_adjust(left=left, right=right, bottom=.20, top=.92, wspace=wspace, hspace=.48)
        for axis, corpus in zip(axes.flat, corpora):
            axis.set_title(LABELS[corpus], fontsize=9)
            axis.tick_params(labelsize=8)
            axis.grid(axis="x", alpha=.18, zorder=0)
        return figure, list(axes.flat)

    def save(figure, name, caption, handles, labels, columns=4):
        legend = figure.legend(handles, labels, loc="lower center", ncol=columns, frameon=False, fontsize=8,
                               columnspacing=.9, handlelength=1.7)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = figure.bbox
        if labels:
            legend_bounds = legend.get_window_extent(renderer)
            for axis in figure.axes:
                if axis.xaxis.label.get_window_extent(renderer).overlaps(legend_bounds):
                    raise ValueError(f"Legend overlaps axis label: {name}")
        check_count = 0
        undrawn_ticks = set()
        for axis in figure.axes:
            for coordinate in (axis.xaxis, axis.yaxis):
                lower, upper = sorted(coordinate.get_view_interval())
                for tick in coordinate.get_major_ticks() + coordinate.get_minor_ticks():
                    if not lower <= tick.get_loc() <= upper:
                        undrawn_ticks.update((tick.label1, tick.label2))
        for text in figure.findobj(matplotlib.text.Text):
            if text not in undrawn_ticks and text.get_visible() and text.get_text():
                box = text.get_window_extent(renderer)
                if box.width > 0 and box.height > 0:
                    if box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
                        raise ValueError(f"Figure text outside print bounds: {name}: {text.get_text()}")
                    check_count += 1
        pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        ink_fraction = float(np.mean(np.any(pixels < 235, axis=2)))
        if not .01 < ink_fraction < .8:
            raise ValueError(f"Blank or invalid figure: {name}")
        for suffix in ("png", "pdf"):
            figure.savefig(out / "figures" / f"{name}.{suffix}", dpi=180, facecolor="white")
        figures.append(dict(name=name, caption=caption, print_width_inches=7.2,
            print_height_inches=float(figure.get_size_inches()[1]), minimum_font_points=8,
            text_bounds_checked=check_count, nonwhite_pixel_fraction=ink_fraction))
        plt.close(figure)

    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        figure, axes = panels()
        handles, labels = [], []
        for axis, corpus in zip(axes, corpora):
            subset = summary[summary["corpus"] == corpus]
            for model in MODELS:
                rows = subset[subset["model"] == model].sort_values("budget_percent")
                finite = np.isfinite(rows["test_perplexity_mean"])
                if finite.any():
                    line = axis.errorbar(rows.loc[finite, "budget_percent"], rows.loc[finite, "test_perplexity_mean"],
                        yerr=rows.loc[finite, "test_perplexity_sd"].fillna(0), color=colors[model],
                        marker="o", markersize=3, linewidth=1, capsize=2,
                        linestyle="--" if model in BACKOFF else "-")
                    if corpus == corpora[0]:
                        handles.append(line)
                        labels.append(names[model])
            axis.set(yscale="log", xlabel="Sentence budget (%)", ylabel="Test PPL (words + EOS)", xticks=BUDGETS)
            lower, upper = axis.get_ylim()
            axis.set_ylim(lower, upper * 3)
            axis.text(.03, .97, "B/T-MLE: infinite\nat every budget", transform=axis.transAxes, va="top", fontsize=8)
        save(figure, "ppl_vs_budget_six_corpora",
             "Test perplexity at five actual sentence-prefix budgets. All seven finite variants are shown; bigram and trigram MLE are infinite at every budget in all six corpora (60 conditions), not clipped to a finite value. PPL includes words and EOS. Error bars are sample SD across neural seeds 13/42/2026 for Brown, Gutenberg, Inaugural and Reuters; ROCStories and WebText have seed 13 only. Vocabulary and model size change with budget, so this is not a fixed-support intervention.",
             handles, labels)
        figure, axes = panels(right=.88, wspace=.80)
        for axis, corpus in zip(axes, corpora):
            rows = budgets[budgets["corpus"] == corpus].sort_values("budget_percent")
            vocabulary_line, = axis.plot(BUDGETS, rows["vocabulary_size_including_specials"] / 1000,
                                         color="#137c64", marker="o", markersize=3)
            secondary = axis.twinx()
            test_line, = secondary.plot(BUDGETS, 100 * rows["test_oov_word_rate"], color="#c66c18", marker="s", markersize=3)
            valid_line, = secondary.plot(BUDGETS, 100 * rows["validation_oov_word_rate"], color="#3269a8", linestyle="--")
            axis.set(xlabel="Sentence budget (%)", ylabel="Vocabulary (thousands)", xticks=BUDGETS)
            secondary.set(ylabel="OOV words (%)", ylim=(0, max(1, 110 * rows[["test_oov_word_rate", "validation_oov_word_rate"]].max().max())))
            secondary.tick_params(labelsize=8)
        save(figure, "vocabulary_oov_by_budget",
             "Actual per-budget vocabulary size (including special tokens) and validation/test OOV rates. Both OOV denominators contain words only, excluding EOS. Vocabularies are built from the selected training prefix; Brown uses minimum frequency 2, the other five corpora use 1. Separate axes show vocabulary in thousands and OOV percentage; curves describe lexical coverage, not a controlled vocabulary-size experiment.",
             [vocabulary_line, test_line, valid_line], ["Vocabulary", "Test OOV", "Validation OOV"], 3)
        figure, axes = panels(6.2)
        for axis, corpus in zip(axes, corpora):
            rows = summary[(summary["corpus"] == corpus) & (summary["budget_percent"] == 100)].set_index("model").loc[MODELS]
            positions = np.arange(len(MODELS))
            overall = axis.barh(positions - .18, 100 * rows["overall_accuracy_mean"], height=.32,
                                xerr=100 * rows["overall_accuracy_sd"].fillna(0), color="#3269a8", capsize=2)
            known = axis.barh(positions + .18, 100 * rows["in_vocabulary_accuracy_mean"], height=.32,
                              xerr=100 * rows["in_vocabulary_accuracy_sd"].fillna(0), color="#137c64", capsize=2)
            value = float(baseline[(baseline["corpus"] == corpus) & (baseline["budget_percent"] == 100)].iloc[0]["always_unk_accuracy"])
            unk_line = axis.axvline(100 * value, color="#c66c18", linestyle="--", linewidth=1.2)
            axis.set(yticks=positions, yticklabels=[names[model] for model in MODELS], xlabel="Word top-1 accuracy (%)")
            axis.invert_yaxis()
        save(figure, "full_budget_accuracy_vs_unk",
             "Full-budget mapped top-1 accuracy over all word targets and raw in-vocabulary word targets for all nine variants. Dashed lines show the always-UNK baseline (mapped UNK targets / all words). OOV correctness means prediction of UNK, not recovery of the original word. EOS is excluded; empty subgroup accuracy remains undefined. Error bars show available sample seed SD only. Laplace unseen-history ties and their attribution are supplied in the accompanying audit table.",
             [overall, known, unk_line], ["Overall words", "In-vocabulary words", "Always UNK"], 3)
        figure, axes = panels(6.2, left=.16, wspace=.80)
        for axis, corpus in zip(axes, corpora):
            rows = summary[(summary["corpus"] == corpus) & (summary["budget_percent"] == 100)].set_index("model")
            count = timing[(timing["corpus"] == corpus) & (timing["budget_percent"] == 100)].iloc[0]
            fit_line = axis.errorbar([count["shared_count_fit_seconds"]], [-.12], color="#137c64", marker="o", linestyle="none", markersize=4)
            for position, model in enumerate(MODELS, start=1):
                row = rows.loc[model]
                test_line = axis.errorbar([row["comparison_test_seconds_mean"]], [position + .12],
                    xerr=[0 if pd.isna(row["comparison_test_seconds_sd"]) else row["comparison_test_seconds_sd"]],
                    color="#3269a8", marker="s", linestyle="none", markersize=3, capsize=2)
                if model in NEURAL:
                    axis.errorbar([row["fit_train_seconds_mean"]], [position - .12],
                        xerr=[0 if pd.isna(row["fit_train_seconds_sd"]) else row["fit_train_seconds_sd"]],
                        color="#137c64", marker="o", linestyle="none", markersize=4, capsize=2)
            axis.set(xscale="log", xlabel="Recorded wall time (seconds)", yticks=np.arange(10),
                     yticklabels=["Shared counts"] + [names[model] for model in MODELS])
            axis.invert_yaxis()
        save(figure, "full_budget_recorded_time",
             "Recorded full-budget fit/train and comparison test wall time, not a hardware-normalized benchmark. The six statistical variants share one counting pass, plotted once as Shared counts; each has a distinct recorded test scoring time. Neural markers are arithmetic seed means with available sample SD. Fit time is metrics.csv train_seconds, reconciled to runs.csv for neural conditions; test time is metrics.csv evaluation_seconds for source=test. Earlier runs.csv test_seconds is separately exported and never added. A recovery allocation JSON records A100-SXM4-80GB / PyTorch 2.11.0+cu126 for the Gutenberg/Inaugural recovery session; it does not establish per-condition hardware, device or utilization, which are unavailable in these CSVs. Current macOS rendering is not experimental runtime provenance.",
             [fit_line, test_line], ["Fit/train (shared counts once)", "Comparison test scoring"], 2)
        figure, axes = panels()
        for axis, corpus in zip(axes, corpora):
            bottom = np.zeros(5)
            category_handles = []
            for category, color in zip(CATEGORIES, category_colors):
                rows = categories[(categories["corpus"] == corpus) & (categories["category"] == CATEGORY_LABELS[category])].drop_duplicates("budget_percent").sort_values("budget_percent")
                bars = axis.bar(BUDGETS, 100 * rows["event_share_including_eos"], bottom=bottom, width=12, color=color)
                category_handles.append(bars)
                bottom += 100 * rows["event_share_including_eos"].to_numpy()
            axis.set(xlabel="Sentence budget (%)", ylabel="Test prediction events (%)", xticks=BUDGETS, ylim=(0, 100))
        save(figure, "test_event_category_shares",
             "Test prediction-event shares from existing audited category CSVs, including EOS. Categories are seen mapped trigram; unseen mapped trigram with seen mapped bigram; unseen mapped bigram with known target; and OOV target. The mutually exclusive shares reconcile to each recorded test-event denominator at every budget. They are mapped-context categories, not necessarily all-known raw histories; event-share OOV is not the word-only OOV rate in the vocabulary figure. Complete conditional PPL means and available seed SD for all nine models are exported without scanning saved scoring archives.",
             category_handles, ["Seen trigram", "Unseen trigram / seen bigram", "Unseen bigram / known target", "OOV target"], 2)
        figure, axes = panels()
        limit = max(abs(loss["weighted_nll_delta_mean"]).max() + loss["weighted_nll_delta_sd"].fillna(0).max(), .01) * 1.15
        for axis, corpus in zip(axes, corpora):
            rows = loss[loss["corpus"] == corpus].set_index("category").loc[CATEGORIES]
            axis.barh(np.arange(4), rows["weighted_nll_delta_mean"], color=category_colors,
                      xerr=rows["weighted_nll_delta_sd"].fillna(0), capsize=2)
            axis.axvline(0, color="black", linewidth=.7)
            axis.set(yticks=np.arange(4), yticklabels=["Seen\ntrigram", "Unseen\ntrigram", "Unseen\nbigram", "OOV\ntarget"],
                     xlabel="Weighted NLL delta", xlim=(-limit, limit))
            axis.set_title(f"{LABELS[corpus]} vs {names[rows.iloc[0]['backoff']]}", fontsize=9)
            axis.invert_yaxis()
        save(figure, "full_budget_loss_decomposition",
             "Full-budget LSTM minus the fixed full-budget best backoff model, decomposed as event share times conditional NLL difference. Negative contributions favor LSTM; positive contributions favor backoff. The same comparator is used for every category within a corpus, and the four contributions sum to the overall per-seed NLL difference (checked against recorded test PPL within saved-event roundoff). Events include EOS. Bars are arithmetic means of per-seed contributions with available sample SD; ROCStories/WebText have one seed. This is an accounting decomposition, not causal attribution or a mixture evaluation.",
             [], [], 1)
    return figures


# ---------------------------------------------------------------- run-level summary

def read_metrics(runs, corpora):
    frames = []
    for corpus in corpora:
        frame = pd.read_csv(runs / RUN_DIRS[corpus] / "metrics" / "metrics.csv")
        frame["corpus"] = corpus
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    metrics["percent"] = metrics["percent"].astype(float).round().astype(int)
    metrics["seed"] = metrics["seed"].apply(lambda v: "" if pd.isna(v) else str(int(float(v))))
    metrics["perplexity"] = metrics["perplexity"].astype(float)
    metrics.loc[metrics["status"] != "finite", "perplexity"] = math.inf
    return metrics


def read_budget_stats(runs, corpora):
    keys = ["train_sentences", "train_tokens", "vocab_size", "valid_oov_rate", "test_oov_rate",
            "test_sentences", "test_tokens"]
    rows = []
    for corpus in corpora:
        for percent in BUDGETS:
            stats = json.loads((runs / RUN_DIRS[corpus] / f"train_{percent}pct" / "data_stats.json").read_text())
            rows.append({"corpus": corpus, "percent": percent, **{key: stats[key] for key in keys}})
    return pd.DataFrame(rows)


def read_neural_runs(runs, corpora):
    frames = []
    for corpus in corpora:
        for percent in BUDGETS:
            frame = pd.read_csv(runs / RUN_DIRS[corpus] / f"train_{percent}pct" / "runs.csv")
            frame["corpus"], frame["percent"] = corpus, percent
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_source(metrics, source):
    rows = []
    for (corpus, percent, model), group in metrics[metrics["source"] == source].groupby(
            ["corpus", "percent", "model"]):
        ppl_mean, ppl_std = mean_std(group["perplexity"])
        acc_mean, acc_std = mean_std(group["accuracy"])
        train_mean, train_std = mean_std(group["train_seconds"].fillna(0))
        rows.append({
            "corpus": corpus, "percent": percent, "model": model, "runs": len(group),
            "ppl_mean": ppl_mean, "ppl_std": ppl_std, "accuracy_mean": acc_mean, "accuracy_std": acc_std,
            "zero_probability_fraction": float((group["zero_probability_events"]
                                                / group["prediction_events"]).mean()),
            "oov_rate": float(group["oov_rate"].mean()),
            "prediction_events": int(group["prediction_events"].iloc[0]),
            "train_seconds_mean": train_mean, "train_seconds_std": train_std,
            "evaluation_seconds_mean": float(group["evaluation_seconds"].mean()),
        })
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], MODELS, ordered=True)
    return frame.sort_values(["corpus", "percent", "model"]).reset_index(drop=True)


def summarize_rare(metrics):
    rows = []
    for (corpus, order, percent, model), group in metrics[metrics["source"] == "rare"].groupby(
            ["corpus", "order", "percent", "model"]):
        mean, std = mean_std(group["perplexity"])
        rows.append({"corpus": corpus, "list": "bigram" if int(order) == 2 else "trigram", "percent": percent,
                     "model": model, "rows": int(group["sentences"].iloc[0]),
                     "events": int(group["prediction_events"].iloc[0]), "rare_ppl_mean": mean,
                     "rare_ppl_std": std})
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], MODELS, ordered=True)
    return frame.sort_values(["corpus", "list", "percent", "model"]).reset_index(drop=True)


def convergence_table(neural_runs):
    rows = []
    for (corpus, percent, variant), group in neural_runs.groupby(["corpus", "percent", "variant"]):
        train_mean, train_std = mean_std(group["train_seconds"])
        rows.append({
            "corpus": corpus, "percent": percent, "model": variant,
            "seeds": ",".join(str(int(s)) for s in sorted(group["seed"])),
            "parameters": int(group["parameters"].iloc[0]),
            "best_epoch": ",".join(str(int(v)) for v in group.sort_values("seed")["best_epoch"]),
            "epochs_trained": ",".join(str(int(v)) for v in group.sort_values("seed")["epochs_trained"]),
            "stop_reason": ",".join(sorted(set(group["stop_reason"]))),
            "train_seconds": fmt_pm(train_mean, train_std),
        })
    return pd.DataFrame(rows)


def ratio_table(test):
    rows = []
    for (corpus, percent), group in test.groupby(["corpus", "percent"]):
        stat = group[group["model"].isin(BACKOFF)].sort_values("ppl_mean").iloc[0]
        neural = group[group["model"].isin(NEURAL)].sort_values("ppl_mean").iloc[0]
        rows.append({"corpus": corpus, "percent": percent,
                     "best_backoff": stat["model"], "best_backoff_ppl": stat["ppl_mean"],
                     "best_neural": neural["model"], "best_neural_ppl": neural["ppl_mean"],
                     "neural_over_backoff": neural["ppl_mean"] / stat["ppl_mean"]})
    return pd.DataFrame(rows)


def write_summary_tables(out, test, valid, rare, budgets, convergence, ratios, corpora):
    tables = out / "tables"
    test.merge(valid[["corpus", "percent", "model", "ppl_mean", "ppl_std"]].rename(
        columns={"ppl_mean": "valid_ppl_mean", "ppl_std": "valid_ppl_std"}),
        on=["corpus", "percent", "model"]).to_csv(tables / "test_metrics_by_budget.csv", index=False)
    budgets.to_csv(tables / "budget_stats.csv", index=False)
    rare.to_csv(tables / "rare_set_ppl.csv", index=False)
    convergence.to_csv(tables / "neural_convergence.csv", index=False)
    ratios.to_csv(tables / "neural_vs_backoff_ratio.csv", index=False)

    params = convergence[convergence["percent"] == 100].set_index(["corpus", "model"])["parameters"]
    main_sections, efficiency_sections, ablation_rows = [], [], []
    for corpus in corpora:
        t100 = test[(test["corpus"] == corpus) & (test["percent"] == 100)]
        v100 = valid[(valid["corpus"] == corpus) & (valid["percent"] == 100)].set_index("model")
        main = pd.DataFrame({
            "Model": t100["model"].astype(str),
            "Test PPL": [fmt_pm(r.ppl_mean, r.ppl_std) for r in t100.itertuples()],
            "Valid PPL": [fmt_pm(v100.loc[r.model, "ppl_mean"], v100.loc[r.model, "ppl_std"])
                          for r in t100.itertuples()],
            "Top-1 acc": [fmt_pm(r.accuracy_mean, r.accuracy_std, 4) for r in t100.itertuples()],
            "Zero-prob events": [f"{r.zero_probability_fraction:.1%}" for r in t100.itertuples()],
            "Params": [f"{params[(corpus, r.model)]:,}" if (corpus, r.model) in params else "-"
                       for r in t100.itertuples()],
            "Train/fit s": [fmt_pm(r.train_seconds_mean, r.train_seconds_std) for r in t100.itertuples()],
            "Test scoring s": [fmt(r.evaluation_seconds_mean) for r in t100.itertuples()],
        })
        stats = budgets[(budgets["corpus"] == corpus) & (budgets["percent"] == 100)].iloc[0]
        seeds = convergence.loc[(convergence["corpus"] == corpus) & (convergence["percent"] == 100),
                                "seeds"].iloc[0]
        if "," in seeds:
            seed_note = f"Neural: mean ± sample std over seeds {seeds}. "
        else:
            seed_note = f"Neural: single seed {seeds}; no estimate of seed variability. "
        note = (f"Train {stats.train_tokens:,} words; vocab {stats.vocab_size:,}; test "
                f"{stats.test_tokens:,} words ({t100['prediction_events'].iloc[0]:,} scored events incl. EOS); "
                f"test OOV {stats.test_oov_rate:.1%}. " + seed_note +
                "N-gram fit time is one shared counting pass for all six n-gram variants.")
        main_sections.append((LABELS[corpus], main, note))

        sub = test[test["corpus"] == corpus]
        efficiency = pd.DataFrame({"Model": MODELS})
        for percent in BUDGETS:
            rows = sub[sub["percent"] == percent].set_index("model")
            efficiency[f"{percent}%"] = [fmt_pm(rows.loc[m, "ppl_mean"], rows.loc[m, "ppl_std"]) for m in MODELS]
        extra = budgets[budgets["corpus"] == corpus].set_index("percent")
        ratio = ratios[ratios["corpus"] == corpus].set_index("percent")
        for label, values in (("train words", [f"{extra.loc[p, 'train_tokens']:,}" for p in BUDGETS]),
                              ("vocab", [f"{extra.loc[p, 'vocab_size']:,}" for p in BUDGETS]),
                              ("test OOV", [f"{extra.loc[p, 'test_oov_rate']:.1%}" for p in BUDGETS]),
                              ("best neural / best backoff",
                               [f"{ratio.loc[p, 'neural_over_backoff']:.3f}" for p in BUDGETS])):
            efficiency.loc[len(efficiency)] = [label, *values]
        efficiency_sections.append((LABELS[corpus], efficiency, None))

        for order in ("bigram", "trigram"):
            for smoothing, label in SMOOTHING.items():
                model = f"{order}_{smoothing}"
                rows = sub[sub["model"] == model].set_index("percent")
                ablation_rows.append({
                    "Corpus": LABELS[corpus], "Order": order, "Smoothing": label,
                    "PPL 20%": fmt(rows.loc[20, "ppl_mean"]), "PPL 100%": fmt(rows.loc[100, "ppl_mean"]),
                    "Zero-prob 20%": f"{rows.loc[20, 'zero_probability_fraction']:.1%}",
                    "Zero-prob 100%": f"{rows.loc[100, 'zero_probability_fraction']:.1%}",
                    "Acc 100%": f"{rows.loc[100, 'accuracy_mean']:.4f}",
                })

    write_md(tables / "main_results_100pct.md", "Main results at 100% training budget", main_sections)
    write_md(tables / "data_efficiency_test_ppl.md", "Test perplexity by training budget",
             [(*section[:2], "Perplexity rises with budget partly because the vocabulary (prediction "
               "support) grows; compare models within a column, not across columns.")
              for section in efficiency_sections])
    write_md(tables / "smoothing_ablation.md", "Smoothing vs no smoothing (test set)",
             [("All corpora", pd.DataFrame(ablation_rows),
               "MLE perplexity is infinite whenever any test event has zero probability.")])
    rare_rows = []
    for corpus in corpora:
        for (rare_list, model), rows in rare[rare["corpus"] == corpus].groupby(["list", "model"], observed=True):
            rows = rows.set_index("percent")
            rare_rows.append({"Corpus": LABELS[corpus], "List": f"{rare_list} ({rows['rows'].iloc[0]:,} n-grams)",
                              "Model": model,
                              **{f"{p}%": fmt_pm(rows.loc[p, "rare_ppl_mean"], rows.loc[p, "rare_ppl_std"])
                                 for p in BUDGETS}})
    write_md(tables / "rare_set_ppl.md", "Perplexity on the isolated rare n-gram lists",
             [("All corpora", pd.DataFrame(rare_rows),
               "Held-out n-grams with zero count in the full training split, each scored with only its own "
               "n-1 word context and weighted by test frequency. Bigram models and ffnn_ctx1 use the bigram "
               "list, trigram models and ffnn_ctx2 the trigram list, the LSTM both; the LSTM loses its usual "
               "sentence prefix here.")])


def plot_summary(out, test, budgets, ratios, runs, corpora):
    figures = out / "figures"
    styles = {m: ("--" if m in BACKOFF else "-") for m in CURVE_MODELS}

    fig, axes = plt.subplots(1, len(corpora), figsize=(5 * len(corpora), 4), squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        tokens = budgets[budgets["corpus"] == corpus].set_index("percent").loc[BUDGETS, "train_tokens"]
        for model in CURVE_MODELS:
            rows = test[(test["corpus"] == corpus) & (test["model"] == model)].set_index("percent").loc[BUDGETS]
            ax.errorbar(tokens, rows["ppl_mean"], yerr=rows["ppl_std"].fillna(0), marker="o", capsize=3,
                        linestyle=styles[model], label=model)
        ax.set(xscale="log", yscale="log", title=LABELS[corpus], xlabel="training words",
               ylabel="test perplexity")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "ppl_vs_train_words.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for corpus in corpora:
        rows = ratios[ratios["corpus"] == corpus].set_index("percent").loc[BUDGETS]
        ax.plot(BUDGETS, rows["neural_over_backoff"], marker="o", label=LABELS[corpus])
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set(xlabel="training budget (%)", ylabel="best neural PPL / best backoff PPL",
           title="Neural advantage (<1 means neural wins)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "neural_vs_backoff_ratio.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(corpora), figsize=(5 * len(corpora), 4), squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        sub = test[test["corpus"] == corpus]
        for model in ("bigram_mle", "trigram_mle"):
            rows = sub[sub["model"] == model].set_index("percent").loc[BUDGETS]
            ax.plot(BUDGETS, 100 * rows["zero_probability_fraction"], marker="o", label=f"{model} zero-prob")
        oov = budgets[budgets["corpus"] == corpus].set_index("percent").loc[BUDGETS, "test_oov_rate"]
        ax.plot(BUDGETS, 100 * oov, marker="s", linestyle=":", color="black", label="test OOV")
        ax.set(title=LABELS[corpus], xlabel="training budget (%)", ylabel="% of test events", ylim=(0, 100))
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "sparsity_vs_budget.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(corpora), figsize=(5 * len(corpora), 4), squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        rows = test[(test["corpus"] == corpus) & (test["percent"] == 100)
                    & np.isfinite(test["ppl_mean"].astype(float))]
        for row in rows.itertuples():
            seconds = max(row.train_seconds_mean, 1e-2)
            ax.scatter(seconds, row.ppl_mean, marker="o" if row.model in NEURAL else "s")
            ax.annotate(str(row.model), (seconds, row.ppl_mean), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set(xscale="log", yscale="log", title=f"{LABELS[corpus]} (100%)", xlabel="train/fit seconds",
               ylabel="test perplexity")
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "runtime_vs_ppl_100pct.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(corpora), figsize=(5 * len(corpora), 4), squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        for model in NEURAL:
            model_dir = runs / RUN_DIRS[corpus] / "train_100pct" / "models" / model
            for history in sorted(model_dir.glob("*_seed13_history.csv")):
                frame = pd.read_csv(history)
                ax.plot(frame["epoch"], frame["valid_ppl"], marker="o", label=f"{model} valid")
                ax.plot(frame["epoch"], frame["train_ppl"], linestyle=":", label=f"{model} train")
        ax.set(yscale="log", title=f"{LABELS[corpus]} (100%, seed 13)", xlabel="epoch", ylabel="perplexity")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "neural_training_curves_100pct.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- event-level analysis

def plot_pipeline(out):
    phases = [
        ("1. Prepare", "Source documents", "prepare_mega_run.freeze_splits", "src/prepare_mega_run.py",
         "Frozen train / valid / test; occurrence provenance", "Document groups; test-first exact quarantine"),
        ("2. Budget + vocabulary", "Frozen train split", "run_full_config; run_from_config", "src/run.py; src/lm/controller.py",
         "train_20pct ... train_100pct; budget vocab.json", "Ordered sentence prefixes; training-only vocabulary"),
        ("3. Neural training", "Budget train + fixed validation", "Trainer.fit", "src/lm/neural/trainer.py",
         "FFNN ctx1/ctx2 and LSTM best weights + histories", "Validation NLL selects checkpoints; test does not"),
        ("4. Count views", "Same budget train + vocabulary", "make_ngram_variants", "src/model_comparison.py",
         "Bigram/trigram MLE, add-one and backoff", "One trigram counting pass; six deterministic views"),
        ("5. Base scoring", "Nine models + fixed split texts", "evaluate_split; write_full_event", "src/model_comparison.py; src/lm/evaluation.py",
         "metrics.csv; scoring_events.jsonl.gz", "PPL includes EOS; word accuracy excludes EOS"),
        ("6. Rare probe", "Saved models + isolated rare n-grams", "evaluate_rare_file", "src/model_comparison.py",
         "Frequency-weighted isolated-sequence results", "Separate diagnostic; not the sentence test population"),
        ("7. Reconcile", "Metrics + event archive + artifact hashes", "verify_scoring_events", "src/model_comparison.py",
         "event_verification.json; artifact_manifest.json", "Verify recorded outputs; no model fitting"),
        ("8. Analyse", "Verified saved results", "run_accuracy_analysis; run_laplace_audit", "src/analyze_comparison_sweep.py",
         "Seed summaries, error strata, UNK audits and figures", "Offline analysis; no retraining or prediction rescoring"),
    ]
    columns = ["phase", "input", "owner", "source", "output", "separation_rule"]
    table = pd.DataFrame(phases, columns=columns)
    for sources in table["source"]:
        for source in sources.split("; "):
            if not (ROOT / source).is_file():
                raise ValueError(f"Pipeline source is missing: {source}")
    figures, tables = out / "figures", out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "pipeline_phases.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, text, color, width=.27, height=.074):
        ax.add_patch(plt.Rectangle((x - width / 2, y - height / 2), width, height,
                                  facecolor=color, edgecolor="#33443f", linewidth=.9))
        ax.text(x, y, text, ha="center", va="center", fontsize=9, linespacing=1.5)

    def arrow(start, end, color="#33443f"):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=1.2))

    ax.text(.5, .975, "Frozen six-corpus experimental pipeline", ha="center", fontsize=15, weight="bold")
    ax.text(.5, .942, "Frozen inputs and budget vocabularies are shared by all model families", ha="center", fontsize=9)
    box(.5, .876, "Source documents\nDocument-level partition + exact quarantine", "#e4eee8", .80)
    arrow((.5, .837), (.5, .801))
    box(.5, .762, "Frozen train / validation / test\nMembership, occurrences and SHA256 manifests", "#e4eee8", .80)
    arrow((.5, .724), (.23, .687))
    arrow((.5, .724), (.76, .687))
    box(.23, .648, "Train: 20/40/60/80/100%\nOrdered sentence prefixes", "#e1f0f1", .39)
    box(.76, .648, "Validation / test fixed\nMapped with each budget vocabulary", "#f3e7e5", .40)
    arrow((.23, .609), (.23, .574))
    box(.23, .534, "Budget-specific vocabulary\nBrown min_freq=2; others 1", "#e1f0f1", .39)
    arrow((.23, .495), (.23, .457))
    arrow((.23, .495), (.76, .457))
    box(.23, .416, "FFNN ctx1 / ctx2 + LSTM\nTrain; select by validation NLL", "#e1f0f1", .39)
    box(.76, .416, "One shared counting pass\nSix n-gram estimator views", "#e1f0f1", .40)
    arrow((.76, .609), (.76, .566), "#9c5956")
    arrow((.76, .566), (.44, .457), "#9c5956")
    ax.text(.65, .566, "validation", ha="center", fontsize=8, color="#9c5956")
    arrow((.96, .648), (.99, .648), "#9c5956")
    arrow((.99, .648), (.99, .302), "#9c5956")
    arrow((.99, .302), (.9, .302), "#9c5956")
    ax.text(.975, .51, "held-out test", rotation=90, ha="center", fontsize=8, color="#9c5956")
    arrow((.23, .377), (.5, .343))
    arrow((.76, .377), (.5, .343))
    box(.5, .302, "Nine-model scoring with the same budget support\nBase split events + separate frequency-weighted rare probe", "#f3e7e5", .80)
    arrow((.5, .263), (.5, .229))
    box(.5, .189, "Saved metrics + complete events + artifact manifests\nReconcile event counts, probabilities and recorded metrics", "#edf0e9", .80)
    arrow((.5, .150), (.5, .116))
    box(.5, .075, "Offline tables, figures and target/history accuracy audits\nNo retraining or prediction rescoring", "#edf0e9", .80)
    fig.savefig(figures / "experimental_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.axis("off")
    ax.set_title("Phase-to-code reference", fontsize=15, pad=18)
    rows = []
    for phase, _, owner, source, _, rule in phases:
        rows.append([phase.replace(" + ", " +\n"), source.replace("; ", "\n") + "\n" + owner.replace("; ", "\n"), rule])
    rendered = ax.table(cellText=rows, colLabels=["Phase", "Owning source / symbol", "Boundary or invariant"],
                        colWidths=[.20, .44, .36], cellLoc="left", loc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8)
    for (row, column), cell in rendered.get_celld().items():
        cell.set_height(.055 if row == 0 else .105)
        cell.set_edgecolor("#c6cec9")
        cell.set_linewidth(.6)
        cell.set_facecolor("#dce8e1" if row == 0 else "white")
        if row == 0:
            cell.set_text_props(weight="bold")
        if column == 2 and row:
            words = cell.get_text().get_text().split()
            lines, current = [], []
            for word in words:
                if len(" ".join(current + [word])) > 35:
                    lines.append(" ".join(current))
                    current = []
                current.append(word)
            lines.append(" ".join(current))
            cell.get_text().set_text("\n".join(lines))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for cell in rendered.get_celld().values():
        text_bounds = cell.get_text().get_window_extent(renderer)
        cell_bounds = cell.get_window_extent(renderer)
        if text_bounds.width > cell_bounds.width or text_bounds.height > cell_bounds.height:
            raise ValueError("Pipeline reference text exceeds its table cell")
    fig.savefig(figures / "phase_code_reference.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote pipeline diagrams and phase map to {out}", flush=True)

def plot_architectures(out):
    summary_root = ROOT / "results/inaugural_100pct_torch_summaries"
    receipt_path = summary_root / "verification.json"
    receipt = json.loads(receipt_path.read_text())
    run = ROOT / receipt["run"]
    vocabulary_size = len(json.loads((run / "vocab.json").read_text()))
    saved_runs = pd.read_csv(run / "runs.csv")
    offsets = {"ffnn_ctx1": 33024, "ffnn_ctx2": 65792, "lstm": 395264}
    records = {record["variant"]: record for record in receipt["models"]}
    if set(records) != set(offsets) or receipt["training_performed"]:
        raise ValueError("Unexpected saved architecture-summary scope")
    rows = []
    for variant, offset in offsets.items():
        record = records[variant]
        expected = 385 * vocabulary_size + offset
        matches = saved_runs[saved_runs["variant"] == variant]
        if (record["vocab_size"] != vocabulary_size or record["total_parameters"] != expected
                or record["trainable_parameters"] != expected
                or not record["strict_checkpoint_load"]
                or len(matches) != 3 or not (matches["parameters"] == expected).all()):
            raise ValueError(f"Saved parameter evidence disagrees for {variant}")
        rows.append({"variant": variant, "vocab_size": vocabulary_size,
                     "total_parameters": expected, "non_embedding_output_parameters": offset,
                     "summary_input": "32 x 10", "summary_output": f"32 x 10 x {vocabulary_size}",
                     "checkpoint_sha256_recorded": record["checkpoint_sha256"],
                     "summary_torch_version": receipt["torch_version"],
                     "summary_torchinfo_version": receipt["torchinfo_version"]})
    figures, tables = out / "figures", out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "neural_architecture_parameters.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    rectangles = []

    def box(x, y, text, color, width=.43, fontsize=9):
        rectangle = plt.Rectangle((x - width / 2, y - .039), width, .078,
                                  facecolor=color, edgecolor="#33443f", linewidth=.9)
        ax.add_patch(rectangle)
        label = ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, linespacing=1.5)
        rectangles.append((rectangle, label))

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="#33443f", linewidth=1.2))

    ax.text(.5, .985, "Neural language-model architectures", ha="center", fontsize=15, weight="bold")
    ax.text(.5, .955, "B: batch; T: padded sequence length; V: stored vocabulary; c: FFNN window", ha="center", fontsize=8)
    box(.5, .895, "Shifted sentence token IDs [B, T]\nFirst input BOS; each output predicts the following target", "#e4eee8", .91)
    for position, title in ((.24, "FFNN: c = 1 or 2"), (.76, "LSTM: sentence prefix")):
        ax.text(position, .832, title, ha="center", fontsize=11, weight="bold", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    arrow((.24, .855), (.24, .807))
    arrow((.76, .855), (.76, .807))
    box(.24, .765, "Causal windows [B, T, c]\nContext 2 starts with PAD / BOS", "#e1f0f1")
    box(.76, .765, "Token embeddings: 128\n[B, T, 128]", "#f3e7e5")
    arrow((.24, .725), (.24, .692))
    arrow((.76, .725), (.76, .692))
    box(.24, .650, "Embeddings + concatenate\n[B, T, 128c]", "#e1f0f1")
    box(.76, .650, "Embedding dropout: 0.3\n[B, T, 128]", "#f3e7e5")
    arrow((.24, .610), (.24, .577))
    arrow((.76, .610), (.76, .577))
    box(.24, .535, "Linear 128c -> 256 + tanh\n[B, T, 256]", "#e1f0f1")
    box(.76, .535, "One LSTM layer: 128 -> 256\n[B, T, 256]; sentence state reset", "#f3e7e5")
    arrow((.24, .495), (.24, .462))
    arrow((.76, .495), (.76, .462))
    box(.24, .420, "Hidden dropout: 0.3\n[B, T, 256]", "#e1f0f1")
    box(.76, .420, "Output dropout: 0.3\n[B, T, 256]", "#f3e7e5")
    arrow((.24, .380), (.24, .347))
    arrow((.76, .380), (.76, .347))
    box(.24, .305, "Linear 256 -> V\nLogits [B, T, V]\nsrc/lm/neural/ffnn/model.py", "#e1f0f1", fontsize=8)
    box(.76, .305, "Linear 256 -> V\nLogits [B, T, V]\nsrc/lm/neural/lstm/model.py", "#f3e7e5", fontsize=8)
    arrow((.24, .265), (.5, .218))
    arrow((.76, .265), (.5, .218))
    box(.5, .175, "Trainer / evaluation: mask PAD and BOS logits\nAllowed targets include UNK and EOS; normalize for scoring", "#e4eee8", .91)
    ax.text(.5, .075, f"Inaugural 100%: V = {vocabulary_size:,}\n"
            f"Parameters: FFNN ctx1 {rows[0]['total_parameters']:,}; ctx2 {rows[1]['total_parameters']:,}; "
            f"LSTM {rows[2]['total_parameters']:,}", ha="center", fontsize=9, linespacing=1.6)
    ax.text(.5, .025, "No future tokens; LSTM context is not capped at four or ten words", ha="center", fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for rectangle, label in rectangles:
        bounds = rectangle.get_window_extent(renderer)
        text = label.get_window_extent(renderer)
        if text.width > bounds.width or text.height > bounds.height:
            raise ValueError("Architecture label exceeds its diagram box")
    fig.savefig(figures / "neural_architectures.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    checks = {"summary_receipt": str(receipt_path.relative_to(ROOT)),
              "summary_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
              "saved_parameter_rows_verified": 9, "model_execution_performed": False,
              "training_performed": False,
              "summary_files_sha256": {f"{variant}.{suffix}": hashlib.sha256(
                  (summary_root / f"{variant}.{suffix}").read_bytes()).hexdigest()
                  for variant in offsets for suffix in ("txt", "html", "png")}}
    (out / "architecture_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Wrote verified architecture diagram and parameter table to {out}", flush=True)


class TestEvents:
    """Per-event test log-probabilities for every (budget, model, seed) condition of one corpus."""

    def __init__(self, path, stride=1, accuracy=False, vocabularies=None, reference=None):
        self.conditions = {}
        self.sentence, self.position, self.eos, self.targets = array("q"), array("q"), bytearray(), []
        reference_key = current_key = None
        logp = flags = None
        next_sentence = next_position = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if '"source": "test"' not in line:
                    continue
                event = json.loads(line)
                if stride > 1 and event["sentence_index"] % stride:
                    continue
                key = (int(round(float(event["percent"]))), event["model"], str(event["seed"]))
                if key != current_key:
                    if key in self.conditions:
                        raise ValueError(f"Scoring events for {key} are not contiguous")
                    current_key, logp, flags = key, array("f"), bytearray()
                    self.conditions[key] = (logp, flags)
                    reference_key = reference_key or key
                if accuracy and key != reference_key:
                    index = len(logp)
                    identity = (event["sentence_index"], event["position"], event["eos"], event["raw_target"])
                    if index >= len(self.targets) or identity != (
                            self.sentence[index], self.position[index], bool(self.eos[index]), self.targets[index]):
                        raise ValueError(f"Test occurrence alignment mismatch for {key} at {index}")
                if accuracy and key == reference_key and reference is not None:
                    if (event["sentence_index"], event["position"]) != (next_sentence, next_position):
                        raise ValueError("Canonical test occurrence order mismatch")
                    words = reference[next_sentence]
                    expected_eos = next_position == len(words)
                    expected_target = "</s>"
                    if not expected_eos:
                        expected_target = words[next_position]
                    if event["raw_target"] != expected_target or event["eos"] != expected_eos:
                        raise ValueError("Canonical test target mismatch")
                    if expected_eos:
                        next_sentence += 1
                        next_position = 0
                    else:
                        next_position += 1
                if accuracy and vocabularies is not None:
                    vocabulary = vocabularies[key[0]]
                    mapping = vocabulary["mapping"]
                    expected_id = mapping.get(event["raw_target"], 1)
                    target = event["target_id"]
                    prediction = event["prediction_id"]
                    if target != expected_id or event["mapped_target"] != vocabulary["words"][target]:
                        raise ValueError("Saved target ID/label disagrees with budget vocabulary")
                    expected_oov = not event["eos"] and event["raw_target"] not in mapping
                    if event["oov"] != expected_oov or event["weight"] != 1:
                        raise ValueError("Saved OOV flag or event weight disagrees with word accounting")
                    if prediction is None:
                        if not key[1].endswith("_mle"):
                            raise ValueError("Only MLE may have no selected prediction")
                    elif prediction in {0, 2} or not 0 <= prediction < len(vocabulary["words"]):
                        raise ValueError("Saved prediction is outside the allowed target support")
                value = event["log_probability"]
                logp.append(-math.inf if value is None else value)
                bits = (ZERO if event["zero_probability"] else 0) | (OOV if event["oov"] else 0)
                if accuracy:
                    bits |= accuracy_flags(event)
                flags.append(bits)
                if key == reference_key:
                    self.sentence.append(event["sentence_index"])
                    self.position.append(event["position"])
                    self.eos.append(1 if event["eos"] else 0)
                    self.targets.append(event["raw_target"])
        size = len(self.targets)
        if reference is not None and (next_sentence != len(reference) or next_position):
            raise ValueError("Saved events do not cover the entire canonical test population")
        for key, (logp, _) in self.conditions.items():
            if len(logp) != size:
                raise ValueError(f"{key} has {len(logp)} test events; expected {size}")
        self.position = np.frombuffer(self.position, dtype=np.int64)
        self.eos = np.frombuffer(bytes(self.eos), dtype=np.uint8).astype(bool)

    def keys(self, percent, model):
        keys = [key for key in self.conditions if key[0] == percent and key[1] == model]
        if not keys:
            raise KeyError(f"No test events for {model} at {percent}%")
        return keys

    def seed_logps(self, percent, model):
        return [np.frombuffer(self.conditions[key][0], dtype=np.float32).astype(np.float64)
                for key in self.keys(percent, model)]

    def logp(self, percent, model):
        return np.mean(self.seed_logps(percent, model), axis=0)

    def flag(self, percent, model, bit):
        flags = np.frombuffer(bytes(self.conditions[self.keys(percent, model)[0]][1]), dtype=np.uint8)
        return (flags & bit).astype(bool)

    def verify(self, metrics, corpus):
        """Recompute perplexity per condition and compare it with metrics.csv."""
        rows = metrics[(metrics["corpus"] == corpus) & (metrics["source"] == "test")]
        for row in rows.itertuples():
            logp = np.frombuffer(self.conditions[(row.percent, row.model, row.seed)][0], dtype=np.float32)
            if len(logp) != row.prediction_events:
                raise ValueError(f"Event count mismatch for {row.percent}/{row.model}/{row.seed}")
            if math.isfinite(row.perplexity):
                ppl = math.exp(-float(np.sum(logp, dtype=np.float64)) / len(logp))
                if not math.isclose(ppl, row.perplexity, rel_tol=1e-4):
                    raise ValueError(f"Perplexity mismatch for {row.percent}/{row.model}/{row.seed}")

    def categories(self, percent):
        oov = self.flag(percent, "bigram_mle", OOV)
        unseen_bigram = self.flag(percent, "bigram_mle", ZERO)
        unseen_trigram = self.flag(percent, "trigram_mle", ZERO)
        return np.select([oov, unseen_bigram, unseen_trigram], ["oov_target", "unseen_bigram", "unseen_trigram"],
                         default="seen_trigram")


def perplexity(logp):
    if len(logp) == 0:
        return math.nan
    if np.isneginf(logp).any():
        return math.inf
    return float(np.exp(-logp.mean()))


def history_membership(train, test, mapping):
    """Mapped history existence, including EOS continuations, without fitting probabilities."""
    one_token, two_token, edges = set(), set(), {}
    next_node = 1
    for words in train:
        previous_two = previous = 2
        node = 0
        for word in words:
            target = mapping.get(word, 1)
            one_token.add(previous)
            two_token.add((previous_two, previous))
            edge = (node, target)
            if edge not in edges:
                edges[edge] = next_node
                next_node += 1
            node = edges[edge]
            previous_two, previous = previous, target
        one_token.add(previous)
        two_token.add((previous_two, previous))
    if not one_token:
        raise ValueError("History membership requires nonempty training sentences")
    masks = {name: bytearray() for name in ("one_token", "two_token", "sentence_prefix")}
    for words in test:
        previous_two = previous = 2
        node = 0
        for position in range(len(words) + 1):
            masks["one_token"].append(previous in one_token)
            masks["two_token"].append((previous_two, previous) in two_token)
            masks["sentence_prefix"].append(node >= 0)
            if position < len(words):
                target = mapping.get(words[position], 1)
                if node >= 0:
                    node = edges.get((node, target), -1)
                previous_two, previous = previous, target
    return {name: np.frombuffer(bytes(values), dtype=np.uint8).astype(bool) for name, values in masks.items()}


def accuracy_population(mask, flags):
    words = int(mask.sum())
    correct = int((mask & ((flags & CORRECT) != 0)).sum())
    predicted_unk = int((mask & ((flags & PREDICTED_UNK) != 0)).sum())
    accuracy = predicted_unk_fraction = math.nan
    if words:
        accuracy = correct / words
        predicted_unk_fraction = predicted_unk / words
    return dict(words=words, correct=correct, accuracy=accuracy, predicted_unk=predicted_unk,
                predicted_unk_fraction=predicted_unk_fraction)


def accuracy_rows(events, corpus, histories):
    conditions, contexts = [], []
    word_mask = ~events.eos
    for key, (_, stored_flags) in events.conditions.items():
        percent, model, seed = key
        flags = np.frombuffer(bytes(stored_flags), dtype=np.uint8)
        oov = (flags & OOV) != 0
        target_unk = (flags & TARGET_UNK) != 0
        groups = {"overall": word_mask, "excluding_unk_targets": word_mask & ~target_unk,
                  "in_vocabulary": word_mask & ~oov, "oov": word_mask & oov,
                  "unk_targets": word_mask & target_unk}
        row = dict(corpus=corpus, percent=percent, model=model, seed=seed)
        for group, mask in groups.items():
            for field, value in accuracy_population(mask, flags).items():
                row[f"{group}_{field}"] = value
        for field, value in accuracy_population(events.eos, flags).items():
            row[f"eos_{field}"] = value
        row["literal_unk_words"] = int((word_mask & target_unk & ~oov).sum())
        if model.startswith("bigram_") or model == "ffnn_ctx1":
            native = histories[percent]["one_token"]
            native_definition = "one mapped preceding token; initial BOS"
        elif model.startswith("trigram_") or model == "ffnn_ctx2":
            native = histories[percent]["two_token"]
            native_definition = "two mapped preceding tokens; initial BOS/BOS"
            if model == "ffnn_ctx2":
                native_definition = "two mapped inputs; initial PAD/BOS (same initial support boolean)"
        else:
            native = histories[percent]["sentence_prefix"]
            native_definition = "exact mapped sentence prefix; initial BOS"
        if model not in NEURAL:
            available = (flags & HISTORY_AVAILABLE) != 0
            seen = (flags & HISTORY_SEEN) != 0
            if not available.all() or not np.array_equal(seen, native):
                raise ValueError(f"Saved statistical history coverage mismatch for {corpus}/{key}")
        elif (flags & HISTORY_AVAILABLE).any():
            raise ValueError("Neural events unexpectedly contain saved count-history coverage")
        row["native_history_definition"] = native_definition
        row["unseen_history_words"] = int((word_mask & ~native).sum())
        row["unseen_history_predicted_unk"] = int((word_mask & ~native & ((flags & PREDICTED_UNK) != 0)).sum())
        row["unseen_history_oov_correct"] = int((word_mask & ~native & oov & ((flags & CORRECT) != 0)).sum())
        conditions.append(row)
        for basis, membership in {"native": native, "one_token": histories[percent]["one_token"],
                                  "two_token": histories[percent]["two_token"]}.items():
            for state, support in (("seen", membership), ("unseen", ~membership)):
                for group in ("overall", "in_vocabulary", "oov"):
                    mask = groups[group] & support
                    contexts.append(dict(corpus=corpus, percent=percent, model=model, seed=seed,
                                         history_basis=basis, history_state=state, target_group=group,
                                         share_of_all_words=float(mask.sum() / word_mask.sum()),
                                         **accuracy_population(mask, flags)))
        if row["overall_words"] != row["in_vocabulary_words"] + row["oov_words"]:
            raise ValueError("Target subgroups do not partition all word events")
        if row["overall_correct"] != row["in_vocabulary_correct"] + row["oov_correct"]:
            raise ValueError("Target subgroup correctness does not reconstruct overall accuracy")
    return conditions, contexts


def accuracy_summary(frame, keys, value_columns):
    for _, group in frame.groupby(keys, dropna=False):
        for column in [name for name in frame.columns if name.endswith("words")]:
            if group[column].nunique() > 1:
                raise ValueError("Seed runs have different subgroup populations")
    return frame.groupby(keys, dropna=False)[value_columns].agg(["mean", "std", "count"])


def laplace_attribution(conditions, contexts, supports):
    baselines, attribution = [], []
    for (corpus, percent), group in conditions.groupby(["corpus", "percent"]):
        if group["overall_words"].nunique() != 1 or group["unk_targets_words"].nunique() != 1:
            raise ValueError("UNK baseline requires identical budget target populations")
        words = int(group.iloc[0]["overall_words"])
        unk_targets = int(group.iloc[0]["unk_targets_words"])
        baselines.append(dict(corpus=corpus, percent=percent, words=words, correct=unk_targets,
                              accuracy=unk_targets / words, allowed_targets=supports[(corpus, percent)]))
        for order in ("bigram", "trigram"):
            model = f"{order}_laplace"
            laplace = group[group["model"] == model].iloc[0]
            mle = group[group["model"] == f"{order}_mle"].iloc[0]
            native = contexts[(contexts["corpus"] == corpus) & (contexts["percent"] == percent)
                              & (contexts["model"] == model) & (contexts["history_basis"] == "native")]
            unseen = native[(native["history_state"] == "unseen") & (native["target_group"] == "overall")].iloc[0]
            unseen_oov = native[(native["history_state"] == "unseen") & (native["target_group"] == "oov")].iloc[0]
            seen = native[(native["history_state"] == "seen") & (native["target_group"] == "overall")].iloc[0]
            if int(unseen["predicted_unk"]) != int(unseen["words"]):
                raise ValueError("Unseen Laplace histories did not all select UNK")
            if int(laplace["overall_correct"]) != int(seen["correct"]) + int(unseen["correct"]):
                raise ValueError("Laplace history partition does not reconstruct accuracy")
            if int(mle["overall_correct"]) != int(seen["correct"]):
                raise ValueError("MLE accuracy differs from Laplace's seen-history contribution")
            if int(mle["in_vocabulary_correct"]) != int(laplace["in_vocabulary_correct"]):
                raise ValueError("Laplace changed in-vocabulary top-1 accuracy")
            allowed = supports[(corpus, percent)]
            attribution.append(dict(corpus=corpus, percent=percent, model=model, words=words,
                                    overall_correct=int(laplace["overall_correct"]),
                                    overall_accuracy=float(laplace["overall_accuracy"]),
                                    in_vocabulary_accuracy=float(laplace["in_vocabulary_accuracy"]),
                                    unseen_words=int(unseen["words"]), unseen_correct=int(unseen["correct"]),
                                    unseen_oov_correct=int(unseen_oov["correct"]),
                                    seen_correct=int(seen["correct"]), seen_predicted_unk=int(seen["predicted_unk"]),
                                    seen_unk_correct=int(laplace["unk_targets_correct"]) - int(unseen["correct"]),
                                    uniform_tie_accuracy_gain=int(unseen["correct"]) / words,
                                    accuracy_abstaining_on_unseen=int(seen["correct"]) / words,
                                    accuracy_random_tie_expected=(int(seen["correct"]) + int(unseen["words"]) / allowed) / words,
                                    always_unk_accuracy=unk_targets / words,
                                    accuracy_minus_unk_baseline=float(laplace["overall_accuracy"]) - unk_targets / words,
                                    allowed_targets=allowed))
    return pd.DataFrame(baselines), pd.DataFrame(attribution)


def count_maxima(counts):
    profiles = {}
    for gram, count in counts.items():
        if count <= 0:
            continue
        context, target = gram[:-1], gram[-1]
        previous = profiles.get(context)
        if previous is None or count > previous[0]:
            profiles[context] = (count, target, 1)
        elif count == previous[0]:
            profiles[context] = (count, min(target, previous[1]), previous[2] + 1)
    return profiles


def verify_laplace_events(runs, corpora, supports, attribution):
    checks = {}
    for corpus in corpora:
        records = {}
        run_dir = runs / RUN_DIRS[corpus]
        brown_percent, profiles = None, {}
        inspected = set()
        with gzip.open(run_dir / "metrics/scoring_events.jsonl.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                if '"source": "test"' not in line:
                    continue
                if not any(f'"model": "{model}"' in line for model in
                           ("bigram_mle", "trigram_mle", "bigram_laplace", "trigram_laplace")):
                    continue
                event = json.loads(line)
                percent, model = int(event["percent"]), event["model"]
                key = (percent, model)
                if key not in records:
                    records[key] = dict(events=0, words=0, correct=0, unseen_words=0, unseen_correct=0,
                                        max_uniform_logp_error=0., seen_digest=hashlib.sha256(),
                                        previous_two=2, previous=2, seen_unk_unique_top_words=0,
                                        seen_unk_tied_top_words=0, seen_unk_unique_top_correct=0,
                                        seen_unk_tied_top_correct=0)
                record = records[key]
                if event["position"] == 0:
                    record["previous_two"] = record["previous"] = 2
                seen = event["context_covered"]
                if not isinstance(seen, bool) or event["weight"] != 1:
                    raise ValueError("Laplace audit requires unit-weight statistical history flags")
                if event["correct"] != (event["target_id"] == event["prediction_id"]):
                    raise ValueError("Saved correctness is inconsistent")
                record["events"] += 1
                if seen:
                    identity = (event["sentence_index"], event["position"], event["raw_target"],
                                event["target_id"], event["prediction_id"])
                    record["seen_digest"].update((json.dumps(identity) + "\n").encode())
                elif model.endswith("_laplace"):
                    expected = -math.log(supports[(corpus, percent)])
                    error = math.inf
                    if event["log_probability"] is not None:
                        error = abs(event["log_probability"] - expected)
                    record["max_uniform_logp_error"] = max(record["max_uniform_logp_error"], error)
                    if event["prediction_id"] != 1 or error > 1e-10:
                        raise ValueError("Unseen Laplace probability/prediction violates the uniform UNK tie rule")
                elif event["prediction_id"] is not None:
                    raise ValueError("Unseen MLE selected a continuation")
                if corpus == "brown" and model.endswith("_laplace"):
                    if brown_percent != percent:
                        budget = run_dir / f"train_{percent}pct"
                        with gzip.open(budget / "count_models.pkl.gz", "rb") as count_handle:
                            bundle = pickle.load(count_handle)
                        vocabulary = json.loads((budget / "vocab.json").read_text())
                        if bundle["schema_version"] != 1 or bundle["vocab"] != vocabulary:
                            raise ValueError("Stored Brown count-table schema/vocabulary mismatch")
                        if bundle["train_sha256"] != hashlib.sha256((budget / "train.txt").read_bytes()).hexdigest():
                            raise ValueError("Stored Brown counts do not match the saved training snapshot")
                        profiles = {}
                        for label in ("bigram_laplace", "trigram_laplace"):
                            view = bundle["variants"][label]["model"]
                            profile = count_maxima(view.counts)
                            if any(view._best_followers.get(context) != values[:2] for context, values in profile.items()):
                                raise ValueError("Stored maximum-count/minimum-ID selection is inconsistent")
                            profiles[label] = profile
                        del bundle
                        brown_percent = percent
                        inspected.add(percent)
                    context = (record["previous"],)
                    if model.startswith("trigram"):
                        context = (record["previous_two"], record["previous"])
                    profile = profiles[model].get(context)
                    if seen != (profile is not None):
                        raise ValueError("Brown saved history coverage disagrees with stored counts")
                    if seen and event["prediction_id"] != profile[1]:
                        raise ValueError("Brown saved prediction disagrees with stored count maximum")
                    if seen and event["prediction_id"] == 1 and not event["eos"]:
                        kind = "unique" if profile[2] == 1 else "tied"
                        record[f"seen_unk_{kind}_top_words"] += 1
                        record[f"seen_unk_{kind}_top_correct"] += int(event["correct"])
                if not event["eos"]:
                    record["words"] += 1
                    record["correct"] += int(event["correct"])
                    if not seen:
                        record["unseen_words"] += 1
                        record["unseen_correct"] += int(event["correct"])
                    record["previous_two"], record["previous"] = record["previous"], event["target_id"]
        if len(records) != 4 * len(BUDGETS):
            raise ValueError("Saved Laplace/MLE condition grid is incomplete")
        selected = attribution[attribution["corpus"] == corpus]
        max_error = 0.
        for index, row in selected.iterrows():
            record = records[(row["percent"], row["model"])]
            mle = records[(row["percent"], row["model"].replace("laplace", "mle"))]
            if record["seen_digest"].digest() != mle["seen_digest"].digest():
                raise ValueError("Saved MLE/Laplace seen-history predictions or identities differ")
            for field in ("words", "correct", "unseen_words", "unseen_correct"):
                target_field = field
                if field == "correct":
                    target_field = "overall_correct"
                if record[field] != int(row[target_field]):
                    raise ValueError("Fresh Laplace event counts disagree with Step 3 attribution")
            if record["events"] != mle["events"]:
                raise ValueError("Saved MLE/Laplace populations differ")
            if corpus != "brown" and int(row["seen_predicted_unk"]):
                raise ValueError("Observed UNK wins outside Brown require additional count inspection")
            for field in ("seen_unk_unique_top_words", "seen_unk_tied_top_words",
                          "seen_unk_unique_top_correct", "seen_unk_tied_top_correct"):
                attribution.loc[index, field] = record[field]
            if record["seen_unk_unique_top_words"] + record["seen_unk_tied_top_words"] != int(row["seen_predicted_unk"]):
                raise ValueError("Unique/tied count maxima do not partition observed UNK selections")
            if record["seen_unk_unique_top_correct"] + record["seen_unk_tied_top_correct"] != int(row["seen_unk_correct"]):
                raise ValueError("Unique/tied count maxima do not partition observed UNK correctness")
            max_error = max(max_error, record["max_uniform_logp_error"])
        checks[corpus] = dict(laplace_conditions=len(selected), seen_mle_prediction_digests_match=True,
                              unseen_uniform_logp_max_error=max_error, brown_count_archives_inspected=len(inspected))
        print(f"{corpus}: {len(selected)} Laplace conditions verified", flush=True)
    return checks


def run_laplace_audit(runs, out, corpora, accuracy_input):
    tables = accuracy_input / "tables"
    conditions = pd.read_csv(tables / "accuracy_by_condition.csv")
    contexts = pd.read_csv(tables / "accuracy_by_context.csv")
    conditions = conditions[conditions["corpus"].isin(corpora)].copy()
    contexts = contexts[contexts["corpus"].isin(corpora)].copy()
    supports = {}
    for corpus in corpora:
        for percent in BUDGETS:
            vocabulary = json.loads((runs / RUN_DIRS[corpus] / f"train_{percent}pct/vocab.json").read_text())
            if vocabulary[:4] != ["<pad>", "<unk>", "<s>", "</s>"] or len(vocabulary) != len(set(vocabulary)):
                raise ValueError("Laplace audit requires the standard unique vocabulary")
            supports[(corpus, percent)] = len(vocabulary) - 2
    baselines, attribution = laplace_attribution(conditions, contexts, supports)
    checks = verify_laplace_events(runs, corpora, supports, attribution)
    comparisons = conditions[["corpus", "percent", "model", "seed", "overall_accuracy", "in_vocabulary_accuracy"]].merge(
        baselines[["corpus", "percent", "accuracy"]].rename(columns={"accuracy": "always_unk_accuracy"}),
        on=["corpus", "percent"], validate="many_to_one")
    comparisons["accuracy_minus_unk_baseline"] = comparisons["overall_accuracy"] - comparisons["always_unk_accuracy"]
    output = out / "tables"
    output.mkdir(parents=True, exist_ok=True)
    baselines.to_csv(output / "always_unk_baseline.csv", index=False)
    attribution.to_csv(output / "laplace_tie_attribution.csv", index=False)
    comparisons.to_csv(output / "accuracy_vs_unk_baseline.csv", index=False)
    source_hashes = {name: hashlib.sha256((tables / name).read_bytes()).hexdigest()
                     for name in ("accuracy_by_condition.csv", "accuracy_by_context.csv")}
    (out / "laplace_checks.json").write_text(json.dumps(dict(corpora=checks, input_table_sha256=source_hashes,
                                                           baseline_rows=len(baselines), laplace_conditions=len(attribution),
                                                           models_fitted=0, neural_checkpoints_loaded=0, predictions_rescored=0,
                                                           counterfactuals="accuracy-only expectations; saved probabilities unchanged"), indent=2) + "\n")
    print(f"Wrote Laplace attribution to {out}", flush=True)


def run_accuracy_analysis(runs, out, corpora, prepared):
    condition_rows, context_rows, checks = [], [], {}
    metrics = read_metrics(runs, corpora)
    for corpus in corpora:
        run_dir = runs / RUN_DIRS[corpus]
        test_path = prepared / corpus / "test.txt"
        test = [line.lower().split() for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        vocabularies = {}
        for percent in BUDGETS:
            budget = run_dir / f"train_{percent}pct"
            stats = json.loads((budget / "data_stats.json").read_text())
            if stats["tokenizer"] != "space" or stats["test_sentences"] != len(test):
                raise ValueError("Accuracy join requires the saved whitespace-tokenized test population")
            vocabulary = json.loads((budget / "vocab.json").read_text())
            if vocabulary[:4] != ["<pad>", "<unk>", "<s>", "</s>"] or len(set(vocabulary)) != len(vocabulary):
                raise ValueError("Invalid budget vocabulary")
            for model in NEURAL:
                variant_vocabulary = json.loads((budget / "models" / model / "vocab.json").read_text())
                if variant_vocabulary != vocabulary:
                    raise ValueError("Neural variant vocabulary differs from the budget vocabulary")
            vocabularies[percent] = dict(words=vocabulary,
                                        mapping={word: index for index, word in enumerate(vocabulary)})
        print(f"{corpus}: streaming saved events for accuracy ...", flush=True)
        events = TestEvents(run_dir / "metrics/scoring_events.jsonl.gz", accuracy=True,
                            vocabularies=vocabularies, reference=test)
        events.verify(metrics, corpus)
        histories = {}
        for percent in BUDGETS:
            train_path = run_dir / f"train_{percent}pct/train.txt"
            train = [line.lower().split() for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            histories[percent] = history_membership(train, test, vocabularies[percent]["mapping"])
            del train
        rows, contexts = accuracy_rows(events, corpus, histories)
        expected = metrics[(metrics["corpus"] == corpus) & (metrics["source"] == "test")]
        actual = {(row["percent"], row["model"], row["seed"]): row for row in rows}
        expected_keys = {(row.percent, row.model, row.seed) for row in expected.itertuples()}
        if len(actual) != len(expected) or set(actual) != expected_keys:
            raise ValueError("Accuracy conditions do not match the saved test metric matrix")
        for saved in expected.itertuples():
            row = actual[(saved.percent, saved.model, saved.seed)]
            for field, value in (("overall_words", saved.word_tokens),
                                 ("overall_correct", saved.correct_predictions), ("oov_words", saved.oov_tokens)):
                if row[field] != int(value):
                    raise ValueError(f"Saved accuracy count mismatch: {corpus}/{saved.model}/{field}")
            if not math.isclose(row["overall_accuracy"], saved.accuracy, abs_tol=1e-12):
                raise ValueError("Reconstructed accuracy disagrees with metrics.csv")
        checks[corpus] = dict(conditions=len(rows), test_words=sum(map(len, test)),
                              test_events=len(events.targets), test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
                              occurrence_alignment=True, vocabulary_and_correctness=True,
                              saved_accuracy_counts=True, statistical_history_membership=True)
        condition_rows.extend(rows)
        context_rows.extend(contexts)
        print(f"{corpus}: {len(rows)} conditions validated", flush=True)
        del events, histories, vocabularies
    conditions, contexts = pd.DataFrame(condition_rows), pd.DataFrame(context_rows)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(tables / "accuracy_by_condition.csv", index=False)
    contexts.to_csv(tables / "accuracy_by_context.csv", index=False)
    values = [column for column in conditions.columns if column.endswith("accuracy") or column.endswith("words")
              or column.endswith("predicted_unk_fraction")]
    summary = accuracy_summary(conditions, ["corpus", "percent", "model"], values)
    summary.columns = [f"{column}_{statistic}" for column, statistic in summary.columns]
    summary.to_csv(tables / "accuracy_summary.csv")
    context_summary = accuracy_summary(contexts, ["corpus", "percent", "model", "history_basis", "history_state", "target_group"],
                                       ["words", "accuracy", "predicted_unk_fraction", "share_of_all_words"])
    context_summary.columns = [f"{column}_{statistic}" for column, statistic in context_summary.columns]
    context_summary.to_csv(tables / "context_accuracy_summary.csv")
    (out / "accuracy_checks.json").write_text(json.dumps(dict(corpora=checks, conditions=len(conditions),
                                                               models_fitted=0, checkpoints_loaded=0,
                                                               accuracy_unit="fraction of word events; EOS reported separately"), indent=2) + "\n")
    print(f"Wrote accuracy evidence to {out}", flush=True)


def subset_ppl(events, percent, model, mask):
    return mean_std([perplexity(logp[mask]) for logp in events.seed_logps(percent, model)])


def category_table(events, percents):
    rows = []
    for percent in percents:
        categories = events.categories(percent)
        for category in CATEGORIES:
            mask = categories == category
            row = {"percent": percent, "category": CATEGORY_LABELS[category], "events": int(mask.sum()),
                   "share": float(mask.mean())}
            for model in MODELS:
                row[f"{model}_ppl"], row[f"{model}_std"] = subset_ppl(events, percent, model, mask)
            rows.append(row)
    return pd.DataFrame(rows)


def cohort_table(events):
    cohort = ~events.flag(20, "bigram_mle", OOV)
    rows = []
    for percent in BUDGETS:
        row = {"percent": percent, "cohort_events": int(cohort.sum()), "cohort_share": float(cohort.mean())}
        for model in MODELS:
            row[f"{model}_ppl"], row[f"{model}_std"] = subset_ppl(events, percent, model, cohort)
        rows.append(row)
    return pd.DataFrame(rows)


def select_cases(events, per_type):
    word = ~events.eos & (events.position >= 2) & np.array([any(c.isalpha() for c in t) for t in events.targets])
    cat20, cat100 = events.categories(20), events.categories(100)
    lp = {model: events.logp(100, model) for model in MODELS}
    lp20 = {model: events.logp(20, model) for model in MODELS}
    best_backoff = np.maximum(lp["bigram_backoff"], lp["trigram_backoff"])
    best_neural = np.maximum.reduce([lp[m] for m in NEURAL])
    best_smoothed = np.maximum.reduce([best_backoff, best_neural, lp["bigram_laplace"], lp["trigram_laplace"]])
    specs = [
        ("unseen trigram: backoff vs add-one", word & (cat100 == "unseen_trigram"),
         lp["trigram_backoff"] - lp["trigram_laplace"]),
        ("unseen bigram: neural beats backoff", word & (cat100 == "unseen_bigram"), lp["lstm"] - best_backoff),
        ("unseen bigram: backoff beats neural", word & (cat100 == "unseen_bigram"), best_backoff - best_neural),
        ("OOV target", word & (cat100 == "oov_target"), best_backoff - lp["lstm"]),
        ("unseen at 20%, seen at 100%", word & (cat20 == "unseen_bigram") & (cat100 == "seen_trigram"),
         lp["lstm"] - lp20["lstm"]),
        ("shared failure (all smoothed models)", word & (cat100 != "oov_target"), -best_smoothed),
    ]
    chosen, used = [], set()
    for name, mask, score in specs:
        candidates = np.flatnonzero(mask & np.isfinite(score))
        picked = 0
        for index in candidates[np.lexsort((candidates, -score[candidates]))]:
            if picked == per_type:
                break
            if int(index) not in used:
                used.add(int(index))
                chosen.append((name, int(index)))
                picked += 1
    rows = []
    for name, index in chosen:
        start = index - int(events.position[index])
        context = events.targets[max(start, index - 6):index]
        row = {"case": name, "sentence_index": int(events.sentence[index]),
               "position": int(events.position[index]), "context": " ".join(context),
               "target": events.targets[index], "category_20": CATEGORY_LABELS[cat20[index]],
               "category_100": CATEGORY_LABELS[cat100[index]]}
        row.update({f"lnp_{model}": float(lp[model][index]) for model in MODELS})
        row.update({"lnp_trigram_backoff_20": float(lp20["trigram_backoff"][index]),
                    "lnp_lstm_20": float(lp20["lstm"][index])})
        rows.append(row)
    return pd.DataFrame(rows)


def add_train_counts(cases, run_dir):
    grams = {}
    for row in cases.itertuples():
        words = row.context.split()[-2:] + [row.target]
        grams[row.Index] = {"uni": (words[-1],), "bi": tuple(words[-2:]), "tri": tuple(words[-3:])}
    wanted = {gram for entry in grams.values() for gram in entry.values()}
    for percent in (20, 100):
        counts = Counter()
        with (run_dir / f"train_{percent}pct" / "train.txt").open(encoding="utf-8") as handle:
            for line in handle:
                tokens = line.lower().split()
                for n in (1, 2, 3):
                    for i in range(len(tokens) - n + 1):
                        gram = tuple(tokens[i:i + n])
                        if gram in wanted:
                            counts[gram] += 1
        for name in ("uni", "bi", "tri"):
            cases[f"train_{name}_{percent}"] = [counts[grams[i][name]] for i in cases.index]
    return cases


def write_event_report(out, corpus, categories, cohort, cases, budgets):
    shown = ["bigram_laplace", "trigram_laplace", *CURVE_MODELS]
    sections = []
    for percent in (20, 100):
        sub = categories[categories["percent"] == percent]
        frame = pd.DataFrame({"Category": sub["category"], "Events": [f"{v:,}" for v in sub["events"]],
                              "Share": [f"{v:.1%}" for v in sub["share"]]})
        for model in shown:
            frame[model] = [fmt_pm(m, s) for m, s in zip(sub[f"{model}_ppl"], sub[f"{model}_std"])]
        sections.append((f"Perplexity by sparsity category at {percent}%", frame,
                         "Category from the same-budget MLE counts: an n-gram is unseen when the MLE model "
                         "assigns it zero probability. MLE columns are omitted (infinite on unseen rows)."))
    frame = pd.DataFrame({"Budget": [f"{p}%" for p in cohort["percent"]],
                          "Cohort events": [f"{v:,}" for v in cohort["cohort_events"]]})
    for model in CURVE_MODELS:
        frame[model] = [fmt_pm(m, s) for m, s in zip(cohort[f"{model}_ppl"], cohort[f"{model}_std"])]
    sections.append(("Fixed cohort: test events whose target is in the 20% vocabulary", frame,
                     "Target occurrence membership is fixed. Prediction support, context mapping, "
                     "training composition and model capacity can still change."))
    case_frame = pd.DataFrame({
        "Case": cases["case"], "Context -> target": [f"... {c} -> **{t}**" for c, t in zip(cases["context"],
                                                                                         cases["target"])],
        "Tri/bi/uni count 20%": [f"{a}/{b}/{c}" for a, b, c in zip(cases["train_tri_20"], cases["train_bi_20"],
                                                                   cases["train_uni_20"])],
        "Tri/bi/uni count 100%": [f"{a}/{b}/{c}" for a, b, c in zip(cases["train_tri_100"],
                                                                    cases["train_bi_100"], cases["train_uni_100"])],
    })
    for model in ("trigram_mle", "trigram_laplace", "trigram_backoff", "ffnn_ctx2", "lstm"):
        case_frame[f"ln p {model}"] = [fmt(v, 2) for v in cases[f"lnp_{model}"]]
    case_frame["ln p lstm 20%"] = [fmt(v, 2) for v in cases["lnp_lstm_20"]]
    sections.append(("Rare/unseen case studies (100% budget)", case_frame,
                     "Raw lowercase surface counts in the exact training subset. Natural-log probabilities; "
                     "neural values are seed means. '-inf' means the model assigned zero probability."))
    write_md(out / "tables" / f"event_analysis_{corpus}.md", f"{LABELS[corpus]}: event-level analysis", sections)

    figures = out / "figures"
    tokens = budgets[budgets["corpus"] == corpus].set_index("percent").loc[BUDGETS, "train_tokens"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in CURVE_MODELS:
        axes[0].errorbar(tokens, cohort[f"{model}_ppl"], yerr=cohort[f"{model}_std"].fillna(0), marker="o",
                         capsize=3, linestyle="--" if model in BACKOFF else "-", label=model)
    axes[0].set(xscale="log", yscale="log", xlabel="training words", ylabel="perplexity on fixed cohort",
                title=f"{LABELS[corpus]}: fixed 20%-vocabulary cohort")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    sub = categories[categories["percent"] == 100]
    width = 0.8 / len(shown)
    for i, model in enumerate(shown):
        axes[1].bar(np.arange(len(sub)) + i * width, sub[f"{model}_ppl"], width, label=model)
    axes[1].set_xticks(np.arange(len(sub)) + 0.4 - width / 2)
    axes[1].set_xticklabels([c.replace(", ", ",\n") for c in sub["category"]], fontsize=8)
    axes[1].set(yscale="log", ylabel="perplexity", title=f"{LABELS[corpus]} (100%): by sparsity category")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / f"event_analysis_{corpus}.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpora", nargs="+", choices=list(RUN_DIRS), default=list(RUN_DIRS))
    parser.add_argument("--events", action="store_true",
                        help="stream scoring_events.jsonl.gz for category, cohort and case analysis (slow)")
    parser.add_argument("--pipeline", action="store_true",
                        help="render documentation-only pipeline and source-reference diagrams")
    parser.add_argument("--architectures", action="store_true",
                        help="render neural architectures from saved summary metadata without model execution")
    parser.add_argument("--final-figures", action="store_true",
                        help="Step 9: final six-corpus figures and metrics from small audited CSV/JSON only")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--accuracy", action="store_true",
                        help="audit saved word accuracy and mapped history membership without model inference")
    parser.add_argument("--laplace-audit", action="store_true",
                        help="verify saved Laplace tie behavior and derive accuracy-only UNK baselines")
    parser.add_argument("--accuracy-input", type=Path, default=ROOT / "results/paper_accuracy_20261107",
                        help="validated accuracy tables used by the Laplace audit")
    parser.add_argument("--prepared", type=Path,
                        default=ROOT / "results/comparison_refresh_20261102_v1/processed",
                        help="canonical prepared test files for accuracy occurrence verification")
    parser.add_argument("--cases-per-type", type=int, default=2)
    args = parser.parse_args()
    if args.final_figures:
        if args.pipeline or args.architectures or args.accuracy or args.laplace_audit or args.events:
            parser.error("Final figures are separate from all other analysis modes")
        output = SUBMISSION_OUT if args.out == DEFAULT_OUT else args.out
        run_submission_figures(args.runs, output, args.corpora, args.accuracy_input)
        return
    if args.architectures:
        if args.pipeline or args.accuracy or args.laplace_audit or args.events:
            parser.error("Architecture rendering is separate from other analysis modes")
        plot_architectures(args.out)
        return
    if args.pipeline:
        if args.accuracy or args.laplace_audit or args.events:
            parser.error("Pipeline rendering is separate from event audit modes")
        plot_pipeline(args.out)
        return
    if args.accuracy and args.laplace_audit:
        parser.error("Choose one audit mode")
    if args.laplace_audit:
        run_laplace_audit(args.runs, args.out, args.corpora, args.accuracy_input)
        return
    if args.accuracy:
        run_accuracy_analysis(args.runs, args.out, args.corpora, args.prepared)
        return
    for folder in ("tables", "figures"):
        (args.out / folder).mkdir(parents=True, exist_ok=True)

    metrics = read_metrics(args.runs, args.corpora)
    budgets = read_budget_stats(args.runs, args.corpora)
    if not args.skip_summary:
        test, valid = summarize_source(metrics, "test"), summarize_source(metrics, "valid")
        ratios = ratio_table(test)
        convergence = convergence_table(read_neural_runs(args.runs, args.corpora))
        write_summary_tables(args.out, test, valid, summarize_rare(metrics), budgets,
                             convergence, ratios, args.corpora)
        plot_summary(args.out, test, budgets, ratios, args.runs, args.corpora)
        print(f"Wrote summary tables and figures to {args.out}", flush=True)
    if args.events:
        for corpus in args.corpora:
            run_dir = args.runs / RUN_DIRS[corpus]
            print(f"{corpus}: streaming test scoring events ...", flush=True)
            events = TestEvents(run_dir / "metrics" / "scoring_events.jsonl.gz")
            events.verify(metrics, corpus)
            print(f"{corpus}: {len(events.targets):,} test events x {len(events.conditions)} conditions "
                  "verified against metrics.csv", flush=True)
            categories, cohort = category_table(events, BUDGETS), cohort_table(events)
            categories.to_csv(args.out / "tables" / f"event_categories_{corpus}.csv", index=False)
            cohort.to_csv(args.out / "tables" / f"fixed_cohort_{corpus}.csv", index=False)
            cases = add_train_counts(select_cases(events, args.cases_per_type), run_dir)
            cases.to_csv(args.out / "tables" / f"rare_unseen_cases_{corpus}.csv", index=False)
            write_event_report(args.out, corpus, categories, cohort, cases, budgets)
            print(f"{corpus}: wrote event analysis ({len(cases)} cases)", flush=True)


if __name__ == "__main__":
    main()
