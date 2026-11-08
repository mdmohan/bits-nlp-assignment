"""Run the remaining frozen v3 corpora on the authorized Kubeflow GPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CORPORA = ("reuters", "roc-stories", "webtext")
PERCENTAGES = (20, 40, 60, 80, 100)


def file_hash(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_plan(workspace, output):
    import yaml

    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    manifest = json.loads((workspace / "bundle_manifest.json").read_text())
    required = {"src/run.py", "src/model_comparison.py", "src/lm/controller.py"}
    for corpus in CORPORA:
        required.add(f"snapshot/configs/{corpus}.yaml")
        for name in ("prepared_manifest.json", "train.txt", "valid.txt", "test.txt"):
            required.add(f"snapshot/processed/{corpus}/{name}")
    files = manifest["files"]
    if not required.issubset(files):
        raise ValueError("Frozen manifest is missing required source or corpus files")
    prefixes = ("src/", "snapshot/configs/") + tuple(
        f"snapshot/processed/{corpus}/" for corpus in CORPORA
    )
    verified = {}
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe frozen member: {name}")
        if name.startswith(prefixes):
            if file_hash(workspace / name) != expected:
                raise ValueError(f"Frozen member checksum mismatch: {name}")
            verified[name] = expected
    configurations = {}
    for corpus in CORPORA:
        frozen = yaml.safe_load((workspace / "snapshot/configs" / f"{corpus}.yaml").read_text())
        if frozen["data"]["corpus"] != corpus:
            raise ValueError(f"Incorrect frozen corpus: {corpus}")
        config = copy.deepcopy(frozen)
        config["device"] = "cuda"
        config["output_dir"] = str(output / "full/runs")
        config.setdefault("execution", {})["colab_only"] = False
        config["data"]["output"] = str(workspace / "snapshot/processed")
        config["data"]["prepared_manifest"] = str(
            workspace / "snapshot/processed" / corpus / "prepared_manifest.json"
        )
        configurations[corpus] = config
    return {
        "corpora": list(CORPORA),
        "percentages": list(PERCENTAGES),
        "workspace": str(workspace),
        "output": str(output),
        "configurations": configurations,
        "verified_files": verified,
        "neural_fits": sum(len(config["seeds"]) * len(config["models"]) * len(PERCENTAGES)
                           for config in configurations.values()),
        "statistical_conditions": len(CORPORA) * len(PERCENTAGES) * 6,
        "execution_adaptation": "Kubeflow CUDA and runtime paths only; frozen settings and sources unchanged",
    }


def verified_reuse(plan, audit_path):
    output = Path(plan["output"]).resolve()
    if json.loads((output / "run_plan.json").read_text()) != plan:
        raise ValueError("Resume protocol differs from the original run")
    if (output / "COMPLETED.json").exists():
        raise ValueError("This run is already complete")
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("read_only") is not True or audit.get("neural_training") is not False:
        raise ValueError("Resume requires a completed read-only neural audit")
    conditions = {}
    for condition in audit["conditions"]:
        key = (condition["corpus"], condition["percent"], condition["variant"], condition["seed"])
        if key in conditions or not condition["complete"] or not condition["weights_usable"]:
            raise ValueError(f"Incomplete or duplicate audited condition: {key}")
        conditions[key] = condition
    reused = {key[0] for key in conditions}
    if not reused or not reused.issubset(CORPORA):
        raise ValueError("Audit must cover completed corpora from this sweep")
    expected = set()
    for corpus in reused:
        config = plan["configurations"][corpus]
        for model in config["models"]:
            variant = model["name"]
            if variant == "ffnn":
                variant = f"ffnn_ctx{model['params']['context_size']}"
            expected.update((corpus, percent, variant, seed)
                            for percent in PERCENTAGES for seed in config["seeds"])
    if set(conditions) != expected:
        raise ValueError("Audit does not cover every required neural condition")
    checked = set()
    for record in audit["records"]:
        if record["stage"] != "neural_training" or not record["required"]:
            continue
        filename = Path(record["path"]).resolve()
        if not filename.is_relative_to(output / "full/runs") or record["status"] != "ok":
            raise ValueError(f"Invalid audited artifact: {filename}")
        if file_hash(filename) != record["sha256"]:
            raise ValueError(f"Audited artifact changed: {filename}")
        if record["kind"] == "final_weights":
            checked.add((record["corpus"], record["percent"], record["variant"], record["seed"]))
    if checked != expected:
        raise ValueError("Audit has no verified weight record for every condition")
    for corpus in set(CORPORA) - reused:
        run_root = output / "full/runs" / plan["configurations"][corpus]["name"]
        if run_root.exists():
            raise ValueError(f"Unaudited existing corpus requires separate recovery: {corpus}")
    return reused


def execute(plan, resume_audit=None):
    import torch
    import yaml
    from packaging.version import Version

    if not os.environ.get("KUBERNETES_SERVICE_HOST") or not torch.cuda.is_available():
        raise RuntimeError("Fitting requires the authorized Kubeflow GPU runtime")
    if Version(str(torch.__version__)).public != "2.11.0":
        raise RuntimeError("Preserve the v3 PyTorch public release 2.11.0")
    workspace, output = Path(plan["workspace"]), Path(plan["output"])
    sys.path.insert(0, str(workspace / "src"))
    from run import run_full_config, verify_model_artifacts, verify_prepared_inputs

    for config in plan["configurations"].values():
        verify_prepared_inputs(config)
    reused = set()
    attempt = output
    if resume_audit is None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    else:
        reused = verified_reuse(plan, resume_audit)
        attempt = output / "resume_attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        attempt.mkdir(parents=True, exist_ok=False)
        (attempt / "resume_plan.json").write_text(json.dumps({
            "audit": str(Path(resume_audit).resolve()), "audit_sha256": file_hash(resume_audit),
            "reuse_neural": sorted(reused), "rerun_evaluation": sorted(reused),
            "new_training": [corpus for corpus in CORPORA if corpus not in reused],
        }, indent=2) + "\n")
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "launcher_sha256": file_hash(__file__),
    }
    (attempt / "runtime_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    config_root = output / "runtime_configs"
    config_root.mkdir(exist_ok=True)
    completed = []
    for corpus in CORPORA:
        config_file = config_root / f"{corpus}.yaml"
        config = plan["configurations"][corpus]
        if config_file.exists():
            if yaml.safe_load(config_file.read_text()) != config:
                raise ValueError(f"Existing runtime configuration differs: {config_file}")
        else:
            config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        if corpus in reused:
            run_root = output / "full/runs" / config["name"]
            backup = attempt / "previous_evaluation" / corpus
            backup.mkdir(parents=True)
            metrics = run_root / "metrics"
            if metrics.exists():
                metrics.rename(backup / "metrics")
            for percent in PERCENTAGES:
                counts = run_root / f"train_{percent}pct/count_models.pkl.gz"
                if counts.exists():
                    destination = backup / f"train_{percent}pct"
                    destination.mkdir()
                    counts.rename(destination / counts.name)
            print(f"RESUME: {corpus}; verified neural checkpoints reused; evaluation only", flush=True)
            subprocess.run([sys.executable, "-u", str(workspace / "src/model_comparison.py"),
                            corpus, "--config", str(config_file), "--device", "cuda"],
                           cwd=workspace, check=True)
            missing = verify_model_artifacts(config_file, config, list(PERCENTAGES), check_comparison=True)
            if missing:
                raise RuntimeError(f"Evaluation output verification failed: {missing[:15]}")
        else:
            print(f"START: {corpus}; neural training followed by full evaluation", flush=True)
            run_full_config(str(config_file), PERCENTAGES, resume=False)
        completed.append(corpus)
        (output / "progress.json").write_text(json.dumps({"completed_corpora": completed}, indent=2) + "\n")
        torch.cuda.empty_cache()
    (output / "COMPLETED.json").write_text(json.dumps({
        "corpora": completed, "neural_fits": plan["neural_fits"],
        "statistical_conditions": plan["statistical_conditions"], "evaluation_complete": True,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume-audit", type=Path,
                        help="Completed neural artifact audit authorizing evaluation-only reuse")
    args = parser.parse_args()
    plan = prepare_plan(args.workspace, args.output)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    execute(plan, args.resume_audit)


if __name__ == "__main__":
    main()