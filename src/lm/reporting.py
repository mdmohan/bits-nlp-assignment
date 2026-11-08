"""Rebuildable, offline reports from saved experiment artifacts."""

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .preparation import atomic_json, file_sha256


MODEL_ORDER = {model: index for index, model in enumerate((
    "bigram_mle", "trigram_mle", "trigram_addk", "lstm", "ffnn",
))}


def _ordered_table(frame):
    if frame.empty:
        return frame

    def sort_key(column):
        if column.name == "model":
            return column.map(MODEL_ORDER).fillna(len(MODEL_ORDER))
        return column

    return frame.sort_values(["corpus", "budget_percent", "model"], key=sort_key, kind="stable")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _training_plot(directory):
    history_path = directory / "history.csv"
    history = pd.read_csv(history_path)
    if history.empty:
        return
    destination = directory / "training.png"
    if destination.exists() and destination.stat().st_mtime_ns >= history_path.stat().st_mtime_ns:
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["epoch"], history["train_ppl"], marker="o", label="Training")
    axis.plot(history["epoch"], history["valid_ppl"], marker="o", label="Validation")
    result = _read(directory / "result.json")
    if result.get("best_epoch") is not None:
        axis.axvline(result["best_epoch"], color="green", linestyle="--", label="Best checkpoint")
        if result["epochs_trained"] != result["best_epoch"]:
            axis.axvline(result["epochs_trained"], color="grey", linestyle=":", label="Last executed")
    axis.set(xlabel="Epoch", ylabel="Perplexity", title=str(directory.parent.name) + " / " + directory.name)
    axis.set_xticks(history["epoch"])
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def _summary(metrics):
    rows = []
    for keys, group in metrics.groupby(["corpus", "budget_percent", "model", "family", "split"], sort=True):
        record = dict(zip(["corpus", "budget_percent", "model", "family", "split"], keys))
        all_finite = group["status"].eq("finite").all()
        record.update(seeds=len(group), finite_seeds=int(group["status"].eq("finite").sum()),
                      ppl_mean=None, ppl_std=None, status="contains_nonfinite")
        if all_finite:
            record.update(ppl_mean=float(group["perplexity"].mean()), ppl_std=0.0, status="finite")
            if len(group) > 1:
                record["ppl_std"] = float(group["perplexity"].std(ddof=1))
        for column in ("accuracy", "oov_rate", "vocab_size", "parameters", "train_seconds", "zero_probability_events", "train_tokens", "process_peak_rss_mb"):
            record[column] = float(group[column].mean())
        for column in ("valid_seconds", "test_seconds", "best_epoch", "epochs_trained",
                       "training_target_events_processed", "training_word_tokens_processed"):
            if column in group:
                record[column] = float(group[column].mean())
        if "convergence_status" in group:
            record["plateau_seeds"] = int(group["convergence_status"].eq("validation_plateau").sum())
            record["budget_limited_seeds"] = int(group["convergence_status"].eq("budget_limited").sum())
        rows.append(record)
    return pd.DataFrame(rows)


def _curve(frame, column, destination, ylabel, title, x_column="budget_percent", uncertainty=None):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for model, points in frame.groupby("model", sort=True):
        points = points.sort_values(x_column)
        values = pd.to_numeric(points[column], errors="coerce")
        axis.plot(points[x_column], values, marker="o", markersize=4, label=model)
        if uncertainty:
            spread = pd.to_numeric(points[uncertainty], errors="coerce")
            axis.fill_between(points[x_column], values - spread, values + spread, alpha=0.12)
    axis.set(ylabel=ylabel, title=title)
    axis.set_xlabel("Training documents (%)")
    if x_column == "train_tokens":
        axis.set_xlabel("Actual training word tokens")
    else:
        axis.set_xticks(list(range(10, 101, 10)))
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def build_report(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    wide_rows = []
    diagnostics = []
    cases = []
    links = []
    for status_path in sorted(output.glob("*/budget_*/final/*/seed*/status.json")):
        status = _read(status_path)
        if status["status"] != "complete":
            continue
        directory = status_path.parent
        if file_sha256(directory / "result.json") != status["artifacts"]["result.json"]:
            raise ValueError(f"Result checksum mismatch while reporting: {directory}")
        result = _read(directory / "result.json")
        relative = directory.relative_to(output).as_posix()
        wide = {key: value for key, value in result.items() if key not in {"documents", "allowed_target_ids"}}
        wide["run_directory"] = relative
        for split, filename in (("valid", "validation_metrics.json"), ("test", "test_metrics.json")):
            path = directory / filename
            if not path.exists():
                continue
            metrics = _read(path)
            if split == "test":
                expected = _read(directory / "evaluation_status.json")["sha256"]
                if file_sha256(path) != expected:
                    raise ValueError(f"Test metrics checksum mismatch while reporting: {directory}")
                wide.update(test_ppl=metrics["perplexity"], test_status=metrics["status"], test_seconds=metrics["test_seconds"])
            row = dict(wide, split=split)
            row.update({key: value for key, value in metrics.items() if key not in {"groups", "cases"}})
            rows.append(row)
            for name, values in metrics["groups"].items():
                diagnostics.append({"corpus": result["corpus"], "budget_percent": result["budget_percent"],
                                    "model": result["model"], "seed": result["seed"], "split": split, "group": name, **values})
            for case in metrics["cases"]:
                cases.append({"run_directory": relative, "split": split, "vocab_hash": result["vocab_hash"], **case})
        wide_rows.append(wide)
        _training_plot(directory)
        description = [f"# {result['model']} / seed {result['seed']}", "",
                       f"Corpus: {result['corpus']}; budget: {result['budget_percent']}%.",
                       f"Training: {result['document_count']} documents, {result['train_sentences']} sentences, {result['train_tokens']} words.",
                       f"Vocabulary: {result['vocab_size']} IDs; hash: {result['vocab_hash']}.",
                       f"Selected setting: {result['selected_value']}; epochs trained: {result['epochs_trained']}.", "",
                       "- [Configuration](config.yaml)", "- [Vocabulary](vocab.json)", "- [Epoch history](history.csv)",
                       "- [Result and provenance](result.json)", "- [Validation metrics](validation_metrics.json)",
                       f"- [Checkpoint]({result['checkpoint']})", f"- [Execution log]({result['attempt']}/run.log)"]
        if result.get("best_epoch") is not None:
            description.extend([
                "",
                f"Restored best epoch: {result['best_epoch']}; last executed epoch: {result['epochs_trained']}.",
                f"Stopping: {result['stop_reason']}; status: {result['convergence_status']}.",
                f"Validation NLL: best {result['best_valid_nll']}; last {result['last_valid_nll']}.",
                "A validation plateau is a stopping-policy outcome, not proof of mathematical convergence.",
            ])
        if (directory / "training.png").exists():
            description.extend(["", "![Training and validation perplexity](training.png)"])
        if (directory / "test_metrics.json").exists():
            description.append("- [Sealed test metrics and examples](test_metrics.json)")
        (directory / "report.md").write_text("\n".join(description) + "\n", encoding="utf-8")
        links.append(f'<li><a href="{html.escape(relative)}/report.md">{html.escape(relative)}</a></li>')
    status_path = output / "corpus_status.json"
    corpus_status = {}
    if status_path.exists():
        corpus_status = _read(status_path)
    protocol_path = output / "protocol.json"
    purpose = "pilot"
    if protocol_path.exists():
        purpose = _read(protocol_path)["config"].get("purpose", "research")
    sections = [f"<h1>Data Efficiency: {html.escape(purpose.title())}</h1>",
                "<p>Per-budget training-only vocabularies change the prediction alphabet. Cross-budget perplexity measures end-to-end growth, not a fixed-target intervention. Infinite scores remain gaps in plots; CSV status and zero-event counts preserve them.</p>"]
    sections.append("<p>Whole-document rounding can repeat subsets at adjacent percentages. See each budget's alias_of_percent and document_count. WebText's approved repair preserves original test bytes and excludes overlapping training source groups; retained raw/cleaned variants keep the original weighting within groups.</p>")
    if purpose == "pilot":
        sections.append("<p><strong>Short-epoch pilot. These results do not establish convergence or final research conclusions.</strong></p>")
    else:
        sections.append("<p>Neural checkpoints are selected on validation loss, not test scores. A patience stop indicates a validation plateau under this policy; a maximum-epoch stop is budget-limited, not established convergence. Seed standard deviations use the sample definition and measure initialization variability on one fixed split, not dataset uncertainty.</p>")
    if rows:
        metrics_frame = _ordered_table(pd.DataFrame(rows))
        summary = _ordered_table(_summary(metrics_frame))
        _ordered_table(pd.DataFrame(wide_rows)).to_csv(output / "runs.csv", index=False)
        metrics_frame.to_csv(output / "metrics.csv", index=False)
        summary.to_csv(output / "summary.csv", index=False)
        _ordered_table(pd.DataFrame(diagnostics)).to_csv(output / "diagnostics.csv", index=False)
        atomic_json(output / "examples.json", cases)
        for corpus, frame in summary.groupby("corpus", sort=True):
            figure_directory = output / corpus / "figures"
            figure_directory.mkdir(parents=True, exist_ok=True)
            split = "valid"
            if frame["split"].eq("test").any():
                split = "test"
            selected = frame.loc[frame["split"] == split]
            plots = [("ppl_mean", "perplexity.png", "Perplexity", "ppl_std"),
                     ("accuracy", "accuracy.png", "Top-1 accuracy", None),
                     ("oov_rate", "oov.png", "OOV word-token rate", None),
                     ("vocab_size", "vocabulary.png", "Vocabulary IDs", None),
                     ("parameters", "parameters.png", "Trainable parameters / count entries", None),
                     ("train_seconds", "runtime.png", "Fit seconds", None),
                     ("zero_probability_events", "zero_events.png", "Zero-probability targets", None),
                     ("process_peak_rss_mb", "memory.png", "Process high-water memory (MiB)", None)]
            sections.append(f"<h2>{html.escape(corpus)} ({split})</h2>")
            for column, filename, label, spread in plots:
                _curve(selected, column, figure_directory / filename, label, f"{corpus}: {split}", uncertainty=spread)
                image = (figure_directory / filename).relative_to(output).as_posix()
                sections.append(f'<img src="{image}" alt="{html.escape(label)}" loading="lazy">')
            _curve(selected, "ppl_mean", figure_directory / "perplexity_words.png", "Perplexity", f"{corpus}: {split}", x_column="train_tokens")
            smoothing = selected.loc[selected["family"] == "ngram"]
            if not smoothing.empty:
                _curve(smoothing, "ppl_mean", figure_directory / "smoothing_ablation.png", "Perplexity", f"{corpus}: smoothing ({split})")
            display_columns = ["budget_percent", "model", "seeds", "finite_seeds", "ppl_mean", "ppl_std", "accuracy", "vocab_size", "oov_rate", "status"]
            display_columns.extend(column for column in (
                "parameters", "train_seconds", "valid_seconds", "test_seconds",
                "plateau_seeds", "budget_limited_seeds",
            ) if column in selected)
            sections.append(selected[display_columns].to_html(index=False, escape=True, na_rep="infinite / unavailable", float_format=lambda value: f"{value:.5g}"))
        run_frame = _ordered_table(pd.DataFrame(wide_rows))
        if "best_epoch" in run_frame:
            neural = run_frame.loc[run_frame["family"] == "neural"]
            columns = [column for column in (
                "corpus", "budget_percent", "model", "seed", "valid_ppl", "test_ppl",
                "best_epoch", "epochs_trained", "stop_reason", "convergence_status",
                "best_valid_nll", "last_valid_nll", "parameters", "train_seconds",
                "valid_seconds", "test_seconds", "training_target_events_processed",
                "training_word_tokens_processed",
            ) if column in neural]
            sections.append("<h2>Neural Checkpoint Selection And Cost</h2>")
            sections.append(neural[columns].to_html(index=False, escape=True, na_rep="unavailable"))
    else:
        sections.append("<p>No completed final runs yet.</p>")
    sections.append("<h2>Corpus Eligibility</h2><pre>" + html.escape(json.dumps(corpus_status, indent=2)) + "</pre>")
    blocked = []
    for path in sorted(output.glob("*/budget_*/budget_status.json")):
        state = _read(path)
        if state["status"] != "complete":
            blocked.append({"budget": path.parent.relative_to(output).as_posix(), **state})
    sections.append("<h2>Incomplete Budgets</h2><pre>" + html.escape(json.dumps(blocked, indent=2)) + "</pre>")
    sections.append("<h2>Run Artifacts</h2><ul>" + "".join(links) + "</ul>")
    style = "body{font:15px 'Segoe UI',sans-serif;color:#202326;background:#fafbfc;margin:24px auto;max-width:1120px;padding:0 16px;letter-spacing:0}h1{font-size:26px}h2{font-size:20px;margin-top:32px}img{width:100%;max-width:540px;height:auto}table{border-collapse:collapse;display:block;overflow:auto;max-width:100%}th,td{padding:6px 10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#12647b}li{overflow-wrap:anywhere}"
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Data Efficiency Results</title><style>' + style + "</style></head><body>" + "\n".join(sections) + "</body></html>"
    (output / "report.html").write_text(document, encoding="utf-8")
    atomic_json(output / "report_metadata.json", {"purpose": purpose, "completed_final_runs": len(wide_rows),
                                                 "corpus_status": corpus_status, "blocked_budgets": blocked,
                                                 "case_count": len(cases), "report": "report.html"})
    print(f"Report: {output / 'report.html'}", flush=True)
    return output / "report.html"