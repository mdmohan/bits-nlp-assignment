"""Recover missing frozen Gutenberg/Inaugural neural artifacts on Kubeflow."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CORPORA = ("gutenberg", "inaugural")
PERCENTAGES = (20, 40, 60, 80, 100)
VARIANTS = ("ffnn_ctx1", "ffnn_ctx2", "lstm")
SEEDS = (13, 42, 2026)


def required_neural_jobs(report):
    if report.get("read_only") is not True or report.get("neural_training") is not False:
        raise ValueError("A completed read-only artifact audit is required")
    expected = {
        (corpus, percent, variant, seed)
        for corpus in CORPORA for percent in PERCENTAGES
        for variant in VARIANTS for seed in SEEDS
    }
    actual = set()
    jobs = []
    for condition in report["conditions"]:
        key = (condition["corpus"], condition["percent"], condition["variant"], condition["seed"])
        if key in actual:
            raise ValueError(f"Duplicate audited condition: {key}")
        actual.add(key)
        if condition["complete"]:
            if not condition["weights_usable"]:
                raise ValueError(f"Contradictory audited condition: {key}")
            continue
        if condition["weights_usable"]:
            raise ValueError(f"Metadata-only damage needs explicit recovery: {key}")
        jobs.append(key)
    if actual != expected:
        raise ValueError("Audit must cover only both corpora, all five budgets, three variants and three seeds")
    return sorted(jobs)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source, destination, expected_hash):
    source, destination = Path(source), Path(destination)
    if not source.is_file() or file_hash(source) != expected_hash:
        raise ValueError(f"Artifact transport checksum mismatch: {source}")
    if destination.exists():
        if file_hash(destination) != expected_hash:
            raise ValueError(f"Refusing to replace a different artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".transport.tmp")
    shutil.copyfile(source, temporary)
    if file_hash(temporary) != expected_hash:
        raise ValueError(f"Copied artifact checksum mismatch: {temporary}")
    temporary.replace(destination)


def runtime_configuration(frozen, workspace, output, variant=None, seed=None):
    from audit_neural_artifacts import expanded_model_specs
    from lm.tokenization import canonical_tokenizer_name

    config = copy.deepcopy(frozen)
    models = expanded_model_specs(frozen)
    config.pop("context_sizes", None)
    config["models"] = models
    config["device"] = "cuda"
    config["output_dir"] = str(Path(output).resolve())
    config.setdefault("execution", {})["colab_only"] = False
    config["data"]["output"] = str(Path(workspace).resolve() / "snapshot/processed")
    config["data"]["prepared_manifest"] = str(Path(config["data"]["output"]) / config["data"]["corpus"] / "prepared_manifest.json")
    config["data"]["tokenizer"] = canonical_tokenizer_name(config["data"].get("tokenizer", "space"))
    if variant is not None:
        selected = []
        for model in models:
            label = model["name"]
            if label == "ffnn":
                label = f"ffnn_ctx{model.get('params', {}).get('context_size', 3)}"
            if label == variant:
                selected.append(model)
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one frozen model for {variant}")
        config["models"] = selected
        config["seeds"] = [seed]
    return config


def rebuild_budget(config, workspace, output, percent, finalize=False):
    import pandas as pd
    import yaml
    from lm.controller import LanguageModelController
    from lm.data import load_corpus_files

    corpus_root = Path(workspace) / "snapshot/processed" / config["data"]["corpus"]
    tokenizer = config["data"]["tokenizer"]
    data = load_corpus_files(corpus_root / "train.txt", corpus_root / "valid.txt", corpus_root / "test.txt",
                             train_percent=percent, tokenizer_name=tokenizer)
    if config["data"].get("lowercase", True):
        data = data.lowercase()
    budget = Path(output) / "full/runs" / config["name"] / f"train_{percent}pct"
    controller = LanguageModelController(data, min_freq=config["vocab"].get("min_freq", 2),
        max_vocab=config["vocab"].get("max_size"), batch_size=config["training"]["batch_size"],
        device=config["device"], output_dir=budget, tokenizer_name=tokenizer, target_support=config.get("target_support"))
    effective = copy.deepcopy(config)
    effective["train_percent"] = percent
    if controller.allowed_target_ids is not None:
        effective["allowed_target_ids"] = list(controller.allowed_target_ids)
    controller.data_stats["train_percent"] = percent
    controller._write_training_subset(corpus_root / "train.txt", budget / "train.txt", len(data.train), tokenizer)
    results = []
    for model in effective["models"]:
        variant = model["name"]
        if variant == "ffnn":
            variant = f"ffnn_ctx{model['params']['context_size']}"
        directory = budget / "models" / variant
        directory.mkdir(parents=True, exist_ok=True)
        variant_config = copy.deepcopy(effective)
        variant_config["models"] = [model]
        variant_config["model_variant"] = variant
        if model["name"] == "ffnn":
            variant_config["context_sizes"] = [model["params"]["context_size"]]
        (directory / "config.yaml").write_text(yaml.safe_dump(variant_config, sort_keys=False))
        (directory / "vocab.json").write_text(json.dumps(controller.vocab.itos))
        (directory / "data_stats.json").write_text(json.dumps(controller.data_stats, indent=2))
        if finalize:
            for seed in effective["seeds"]:
                result_file = directory / f"{model['name']}_seed{seed}_result.json"
                results.append(json.loads(result_file.read_text())["result"])
    (budget / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False))
    (budget / "vocab.json").write_text(json.dumps(controller.vocab.itos))
    (budget / "data_stats.json").write_text(json.dumps(controller.data_stats, indent=2))
    if finalize:
        runs = pd.DataFrame(results)
        groups = ["variant", "model"]
        if "context_size" in runs.columns:
            groups.append("context_size")
        runs.to_csv(budget / "runs.csv", index=False)
        runs.groupby(groups, dropna=False)[["valid_ppl", "test_ppl", "train_seconds", "test_seconds"]].agg(["mean", "std"]).to_csv(budget / "summary.csv")


def recover(workspace, retained, audit_path, output):
    import torch
    import yaml
    from packaging.version import Version

    workspace, retained, output = Path(workspace).resolve(), Path(retained).resolve(), Path(output).resolve()
    if not os.environ.get("KUBERNETES_SERVICE_HOST") or not torch.cuda.is_available():
        raise RuntimeError("Neural fitting requires the authorized Kubeflow GPU runtime")
    if Version(str(torch.__version__)).public != "2.11.0":
        raise RuntimeError("Preserve the frozen PyTorch public release 2.11.0")
    sys.path.insert(0, str(workspace / "src"))
    from audit_neural_artifacts import ArtifactAudit
    from lm.controller import run_from_config
    from run import verify_prepared_inputs

    report = json.loads(Path(audit_path).read_text())
    jobs = required_neural_jobs(report)
    output.mkdir(parents=True, exist_ok=True)
    plan = {"corpora": list(CORPORA), "percentages": list(PERCENTAGES), "jobs": [list(job) for job in jobs],
            "reused_conditions": 90 - len(jobs), "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0), "platform": "kubeflow",
            "execution_adaptation": "runtime configurations disable colab_only; frozen sources/settings unchanged"}
    plan_file = output / "recovery_plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Existing recovery output has a different protocol")
    plan_file.write_text(json.dumps(plan, indent=2))
    original_root = Path("/content/drive/MyDrive/NLP/comparison_sweep_20261102_v3")
    complete = {(item["corpus"], item["percent"], item["variant"], item["seed"]) for item in report["conditions"] if item["complete"]}
    carried_records = []
    for record in report["records"]:
        key = (record["corpus"], record["percent"], record["variant"], record["seed"])
        if key in complete and record["kind"] in ("final_weights", "history", "signed_result"):
            carried_records.append(record)
    configurations = {}
    config_root = output / "runtime_configs"
    config_root.mkdir(exist_ok=True)
    for corpus in CORPORA:
        frozen = yaml.safe_load((workspace / "snapshot/configs" / f"{corpus}.yaml").read_text())
        config = runtime_configuration(frozen, workspace, output / "full/runs")
        verify_prepared_inputs(config)
        configurations[corpus] = config
        (config_root / f"{corpus}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        for percent in PERCENTAGES:
            rebuild_budget(config, workspace, output, percent)
    for index, (corpus, percent, variant, seed) in enumerate(jobs, 1):
        print(f"NEURAL JOB {index}/{len(jobs)}: {corpus} {percent}% {variant} seed {seed}", flush=True)
        job_root = output / "jobs" / f"{corpus}_{percent}_{variant}_{seed}"
        job_root.mkdir(parents=True, exist_ok=True)
        config = runtime_configuration(configurations[corpus], workspace, job_root / "runs", variant, seed)
        config_file = job_root / "config.yaml"
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        model_name = config["models"][0]["name"]
        directory = job_root / "runs" / config["name"] / f"train_{percent}pct" / "models" / variant
        stem = f"{model_name}_seed{seed}"
        if (directory / f"{stem}_result.json").exists():
            for suffix in (".pt", "_history.csv", "_result.json"):
                artifact = directory / f"{stem}{suffix}"
                if not artifact.is_file() or not artifact.stat().st_size:
                    raise ValueError(f"Refusing unsafe result-only resume: {artifact}")
        run_from_config(str(config_file), train_percent=percent, prepare_data=False, resume=True)
        destination = output / "full/runs" / config["name"] / f"train_{percent}pct" / "models" / variant
        for suffix in (".pt", "_history.csv", "_result.json"):
            source = directory / f"{stem}{suffix}"
            copy_verified(source, destination / source.name, file_hash(source))
        (output / "progress.json").write_text(json.dumps({"completed_neural_jobs": index, "total_neural_jobs": len(jobs), "last": [corpus, percent, variant, seed]}, indent=2))
        torch.cuda.empty_cache()
    missing_carried = [str(Path(record["path"]).relative_to(original_root)) for record in carried_records
                      if not (retained / Path(record["path"]).relative_to(original_root)).is_file()]
    if missing_carried:
        (output / "WAITING_FOR_RETAINED.json").write_text(json.dumps({"neural_jobs_completed": len(jobs), "missing_files": missing_carried}, indent=2))
        print("NEURAL PHASE COMPLETE; verified retained artifacts are still required before evaluation", flush=True)
        return
    for record in carried_records:
        relative = Path(record["path"]).relative_to(original_root)
        copy_verified(retained / relative, output / relative, record["sha256"])
    (output / "WAITING_FOR_RETAINED.json").unlink(missing_ok=True)
    for corpus, config in configurations.items():
        for percent in PERCENTAGES:
            rebuild_budget(config, workspace, output, percent, finalize=True)
    verification = ArtifactAudit(workspace, output, output / "pre_evaluation_audit").run()
    if not all(item["complete"] for item in verification["conditions"]):
        raise ValueError("Complete neural artifacts are required before comparison")
    for corpus in CORPORA:
        print(f"EVALUATION: {corpus}, all five budgets", flush=True)
        subprocess.run([sys.executable, "-u", str(workspace / "src/model_comparison.py"), corpus,
                        "--config", str(config_root / f"{corpus}.yaml"), "--device", "cuda"], cwd=workspace, check=True)
    final = ArtifactAudit(workspace, output, output / "final_audit").run()
    if any(record["required"] and record["status"] != "ok" for record in final["records"]):
        raise ValueError("Final artifact audit found incomplete outputs")
    (output / "COMPLETED.json").write_text(json.dumps({"neural_conditions": 90, "new_fits": len(jobs), "reused_fits": len(complete), "corpora": list(CORPORA), "evaluation_complete": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(required_neural_jobs(json.loads(args.audit.read_text())), indent=2))
        return
    recover(args.workspace, args.retained, args.audit, args.output)


if __name__ == "__main__":
    main()