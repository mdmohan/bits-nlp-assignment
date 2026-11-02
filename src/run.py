"""Training and end-to-end corpus runner.

Examples::

    python src/run.py src/configs/roc-stories.yaml --train-percent 20
    python src/run.py --full src/configs/mega-run.yaml
    python src/run.py --full src/configs/mega-run.yaml --parallelism 2
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import yaml

from lm.controller import run_from_config
from lm.tokenization import canonical_tokenizer_name, tokenizer_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERCENTAGES = [20, 40, 60, 80, 100]


def percentage(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 100") from exc
    if not 0 < parsed <= 100:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 100")
    return parsed


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        alternate = PROJECT_ROOT / "src" / "configs" / path.name
        if alternate.is_file():
            path = alternate
    return path.resolve()


def load_full_manifest(path: Path) -> tuple[list[Path], list[float]]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Full-run config must be a YAML mapping: {path}")
    configured = manifest.get("run", manifest.get("configs"))
    if not isinstance(configured, list) or not configured:
        raise ValueError("Full-run config must have a nonempty 'run: [config.yaml, ...]' list")
    percentages = manifest.get("train_percentages", manifest.get("percentages", DEFAULT_PERCENTAGES))
    if (not isinstance(percentages, list) or not percentages
            or any(not isinstance(item, (int, float)) or not 0 < item <= 100 for item in percentages)):
        raise ValueError("train_percentages must be a nonempty list of numbers in (0, 100]")
    percentages = [float(value) for value in percentages]
    if sorted(set(percentages)) != percentages:
        raise ValueError("train_percentages must be unique and strictly increasing")
    configs = []
    for item in configured:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Every item in the full-run 'run' list must be a YAML filename")
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            local = path.parent / candidate
            candidate = local if local.is_file() else PROJECT_ROOT / candidate
            if not candidate.is_file():
                fallback = PROJECT_ROOT / "src/configs" / Path(item).name
                if fallback.is_file():
                    candidate = fallback
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Experiment config not found: {item}")
        configs.append(candidate)
    return configs, percentages


def expected_variants(config: dict) -> list[tuple[str, str, int | str]]:
    """Match the model-variant folder names used by run_from_config."""
    configured_models = [
        {**model, "params": dict(model.get("params", {}))}
        for model in config.get("models", [])
    ]
    context_sizes = config.get("context_sizes")
    expanded = []
    for model in configured_models:
        if str(model.get("name", "")).casefold() == "ffnn" and context_sizes:
            for context_size in context_sizes:
                variant = {**model, "params": dict(model["params"])}
                variant["params"]["context_size"] = context_size
                expanded.append(variant)
        else:
            expanded.append(model)
    counts: dict[str, int] = {}
    output = []
    for model in expanded:
        model_name = str(model["name"])
        if model_name.casefold() == "ffnn":
            context_size = model.get("params", {}).get("context_size", 3)
            label = f"ffnn_ctx{context_size}"
        else:
            label = model_name
        counts[label] = counts.get(label, 0) + 1
        folder = label if counts[label] == 1 else f"{label}_{counts[label]}"
        output.append((folder, model_name, context_size if model_name.casefold() == "ffnn" else ""))
    return output


def verify_model_artifacts(config_path: Path, config: dict, percentages: list[float],
                           check_comparison: bool = False) -> list[str]:
    missing = []
    output_root = Path(config.get("output_dir", "results"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_root = output_root / config["name"]
    seeds = config.get("seeds", [42])
    variants = expected_variants(config)
    corpus = config["data"]["corpus"]
    for percent in percentages:
        directory = run_root / f"train_{percent:g}pct"
        if not (directory / "train.txt").is_file():
            missing.append(str(directory / "train.txt"))
        for folder, model_name, _ in variants:
            for seed in seeds:
                expected = directory / "models" / folder / f"{model_name}_seed{seed}.pt"
                if not expected.is_file():
                    missing.append(str(expected))
    if missing:
        return missing

    if check_comparison:
        comparison_root = run_root / "metrics"
        required = [
            comparison_root / "metrics.csv",
            comparison_root / "charts.png",
            comparison_root / "predictions.csv",
            comparison_root / "predictions_correct.csv",
            comparison_root / "predictions_incorrect.csv",
            comparison_root / "perplexity_by_percentage.csv",
        ]
        missing.extend(str(path) for path in required if not path.is_file() or path.stat().st_size == 0)
        for percent in percentages:
            plot_percent = f"{percent:g}".replace(".", "_")
            plot_path = comparison_root / f"plot_{plot_percent}p.png"
            if not plot_path.is_file() or plot_path.stat().st_size == 0:
                missing.append(str(plot_path))
        metrics_path = comparison_root / "metrics.csv"
        if metrics_path.is_file():
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            observed = {
                (row.get("model"), row.get("seed", ""), row.get("source"), float(row["percent"]))
                for row in rows if row.get("percent")
            }
            expected_labels = [(folder, str(seed)) for folder, _, _ in variants for seed in seeds]
            expected_labels += [
                (label, "") for label in (
                    "bigram_mle", "bigram_laplace", "bigram_backoff",
                    "trigram_mle", "trigram_laplace", "trigram_backoff",
                )
            ]
            for percent in percentages:
                for label, seed in expected_labels:
                    for source in ("train", "valid", "test"):
                        if (label, seed, source, percent) not in observed:
                            missing.append(f"metrics row: {percent:g}% {label} seed={seed} {source}")
    return missing


def run_full_config(config_path: str, percentages: tuple[float, ...], resume: bool = False) -> dict:
    project_root = PROJECT_ROOT
    os.chdir(project_root)
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError(f"Invalid corpus experiment config: {config_path}")
    corpus = config["data"].get("corpus")
    if not corpus:
        raise ValueError(f"{config_path.name} has no data.corpus")
    print(f"\n===== FULL PIPELINE: {corpus} ({config_path.name}) =====", flush=True)

    orchestrator = project_root / "src/data-pipeline/data_orchestrator.py"
    tokenizer = canonical_tokenizer_name(config["data"].get("tokenizer", "space"))
    print(f"[{corpus}] Prepare/load cached corpus once (use_cached={config['data'].get('use_cached', True)})", flush=True)
    subprocess.run(
        [sys.executable, str(orchestrator), "--config", str(config_path), "--token", tokenizer],
        cwd=project_root,
        check=True,
    )

    for percent in percentages:
        print(f"\n[{corpus}] Train all configured models at {percent:g}%", flush=True)
        run_from_config(
            str(config_path),
            train_percent=percent,
            resume=resume,
            prepare_data=False,
        )
        missing = verify_model_artifacts(config_path, config, [percent])
        if missing:
            raise RuntimeError(f"Model outputs missing for {corpus} at {percent:g}%: {missing[:10]}")
        print(f"[{corpus}] Verified model checkpoints and train snapshot for {percent:g}%.", flush=True)

    comparison = project_root / "src/model_comparison.py"
    print(f"\n[{corpus}] Compare all percentage runs", flush=True)
    subprocess.run([sys.executable, str(comparison), corpus], cwd=project_root, check=True)
    missing = verify_model_artifacts(config_path, config, list(percentages), check_comparison=True)
    if missing:
        raise RuntimeError(f"Full result verification failed for {corpus}: {missing[:15]}")
    print(f"[{corpus}] Full pipeline complete; all expected outputs verified.", flush=True)
    return {"corpus": corpus, "config": str(config_path), "percentages": list(percentages)}


def run_full(manifest_path: Path, parallelism: int, resume: bool = False) -> None:
    configs, percentages = load_full_manifest(manifest_path)
    print(f"Full run: {len(configs)} corpus configs; percentages={percentages}; parallelism={parallelism}")
    if parallelism == 1:
        for config_path in configs:
            run_full_config(str(config_path), tuple(percentages), resume)
        return

    failures = []
    with ProcessPoolExecutor(max_workers=parallelism, mp_context=get_context("spawn")) as pool:
        futures = {
            pool.submit(run_full_config, str(config_path), tuple(percentages), resume): config_path
            for config_path in configs
        }
        for future in as_completed(futures):
            config_path = futures[future]
            try:
                result = future.result()
                print(f"Completed {result['corpus']}.", flush=True)
            except Exception as exc:
                failures.append((config_path, exc))
                print(f"Failed {config_path.name}: {exc}", flush=True)
    if failures:
        details = "; ".join(f"{path.name}: {error}" for path, error in failures)
        raise RuntimeError(f"Full pipeline failed: {details}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data, train language models, and compare results.")
    parser.add_argument("config", nargs="?", help="path to a legacy YAML experiment config")
    parser.add_argument("--full", metavar="MEGA_CONFIG", help="run prepare, all configured train percentages, and model comparison for each listed corpus")
    parser.add_argument("--parallelism", type=positive_int, default=1,
                        help="number of corpus pipelines to run in separate processes (default: 1; requires --full)")
    parser.add_argument("--suite", help="path to a prepared-document sweep YAML config")
    parser.add_argument("--stage", choices=("preflight", "prepare", "train", "seal", "evaluate", "report", "all"), default="preflight")
    parser.add_argument("--corpora", nargs="+", help="restrict suite execution to named corpora")
    parser.add_argument(
        "--resume", action="store_true",
        help="resume interrupted neural fits from epoch checkpoints; for --suite, reuse verified fits",
    )
    parser.add_argument("--dry-run", action="store_true", help="display suite matrix without starting work")
    parser.add_argument(
        "--train-percent",
        type=percentage,
        default=100,
        metavar="PERCENT",
        help="use the first PERCENT of train.txt sentences (default: 100)",
    )
    parser.add_argument(
        "--algorithm",
        "--model",
        dest="algorithm",
        help="run one configured algorithm (for example: ffnn or lstm); default: all configured models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="override configured seeds and run once with this seed",
    )
    parser.add_argument(
        "--context-size",
        "--ffnn-context-size",
        dest="context_size",
        type=int,
        metavar="WORDS",
        help=(
            "override FFNN context size (number of preceding words); "
            "1 is bigram-style, 2 is trigram-style"
        ),
    )
    parser.add_argument(
        "--token",
        choices=tokenizer_names(),
        default=None,
        help="tokenizer: space (default), wp (WordPunct), or tb (Treebank)",
    )
    args = parser.parse_args()
    if args.full:
        if (args.config or args.suite or args.corpora or args.dry_run or args.stage != "preflight"
                or args.algorithm or args.seed is not None or args.context_size is not None
                or args.token is not None or args.train_percent != 100):
            parser.error("--full cannot be combined with positional configs, --suite, or per-run overrides")
        run_full(resolve_path(args.full), args.parallelism, resume=args.resume)
    elif args.suite:
        if args.parallelism != 1:
            parser.error("--parallelism requires --full")
        if (args.config or args.train_percent != 100 or args.algorithm or args.seed is not None
                or args.context_size is not None or args.token is not None):
            parser.error("Suite settings belong in its YAML; legacy runner overrides cannot be combined with --suite")
        from lm.experiments import ExperimentSuite
        ExperimentSuite(args.suite, args.corpora).run(args.stage, resume=args.resume, dry_run=args.dry_run)
    else:
        if args.parallelism != 1:
            parser.error("--parallelism requires --full")
        if not args.config:
            parser.error("Provide a legacy config or --suite")
        if args.corpora or args.dry_run or args.stage != "preflight":
            parser.error("--stage, --corpora and --dry-run require --suite")
        run_from_config(
            args.config,
            train_percent=args.train_percent,
            algorithm=args.algorithm,
            seed_override=args.seed,
            context_size_override=args.context_size,
            tokenizer_override=args.token,
            resume=args.resume,
        )
