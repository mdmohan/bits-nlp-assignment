"""Back up completed neural training artifacts without reading live metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from run_kubeflow_recovery import file_hash, required_neural_jobs


def export_models(task_root, output):
    root = Path(task_root).resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(root / "neural_exports"):
        raise ValueError("Export must use a new task-local neural_exports directory")
    audit_path = root / "recovered_run/pre_evaluation_audit/audit.json"
    report = json.loads(audit_path.read_text())
    if required_neural_jobs(report):
        raise ValueError("All 90 neural conditions must be complete before backup")
    records = [record for record in report["records"]
               if record["required"] and record["stage"] == "neural_training"]
    expected = {"budget_artifact": 60, "variant_config": 30,
                "variant_vocabulary": 30, "variant_data_stats": 30,
                "final_weights": 90, "history": 90, "signed_result": 90}
    if dict(Counter(record["kind"] for record in records)) != expected:
        raise ValueError("Unexpected neural artifact matrix")
    entries = {}
    for record in records:
        source = Path(record["path"]).resolve()
        source.relative_to(root / "recovered_run/full/runs")
        if record["status"] != "ok" or not record.get("sha256"):
            raise ValueError(f"Unverified neural artifact: {source}")
        relative = source.relative_to(root).as_posix()
        if relative in entries:
            raise ValueError(f"Duplicate neural artifact: {relative}")
        entries[relative] = (source, record["sha256"])
    for source in sorted((root / "frozen_workspace").rglob("*")):
        if source.is_file():
            source.resolve().relative_to(root / "frozen_workspace")
            entries[source.relative_to(root).as_posix()] = (source, None)
    provenance = [
        "source_audit.json", "source_transport_manifest.json",
        "retained_import_receipt.json", "retained_transport_receipt.json",
        "runner_cuda_update_receipt.json", "runner_update_receipt.json",
        "nlp_environment.json", "dependency_repair.json",
        "extras/run_kubeflow_recovery.py", "extras/audit_neural_artifacts.py",
        "recovered_run/runtime_environment.json", "recovered_run/recovery_plan.json",
        "recovered_run/progress.json", "recovered_run/pre_evaluation_audit/audit.json",
        "recovered_run/pre_evaluation_audit/artifact_status.csv",
        "recovered_run/pre_evaluation_audit/MISSING_ARTIFACTS.md",
        "recovered_run/runtime_configs/gutenberg.yaml",
        "recovered_run/runtime_configs/inaugural.yaml",
    ]
    for relative in provenance:
        source = root / relative
        if source.is_file():
            entries[relative] = (source, None)
    output.mkdir(parents=True, exist_ok=False)
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    if shutil.which("ionice"):
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())],
                       capture_output=True, check=False)
    archive_path = output / "neural_models.zip"
    temporary = output / "neural_models.zip.part"
    manifest = {"schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "neural_conditions": 90, "neural_training_files": len(records),
                "audit_sha256": file_hash(audit_path),
                "evaluation_outputs_included": False, "files": []}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as archive:
        for relative, (source, expected_hash) in sorted(entries.items()):
            before = source.stat()
            digest = hashlib.sha256()
            copied = 0
            with source.open("rb") as handle, archive.open(relative, "w", force_zip64=True) as target:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    target.write(block)
                    digest.update(block)
                    copied += len(block)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError(f"Source changed during backup: {relative}")
            actual_hash = digest.hexdigest()
            if copied != before.st_size or (expected_hash and actual_hash != expected_hash):
                raise ValueError(f"Backup checksum mismatch: {relative}")
            manifest["files"].append({"path": relative, "bytes": copied,
                                      "sha256": actual_hash})
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        archive.writestr("neural_models_manifest.json", manifest_bytes)
    with zipfile.ZipFile(temporary) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != len(entries) + 1:
            raise ValueError("Invalid backup archive membership")
        for item in manifest["files"]:
            if archive.getinfo(item["path"]).file_size != item["bytes"]:
                raise ValueError(f"Invalid archive size: {item['path']}")
    temporary.rename(archive_path)
    (output / "neural_models_manifest.json").write_bytes(manifest_bytes)
    archive_hash = file_hash(archive_path)
    receipt = {"archive": archive_path.name, "bytes": archive_path.stat().st_size,
               "sha256": archive_hash, "neural_conditions": 90,
               "neural_training_files": len(records), "files": len(entries),
               "evaluation_outputs_included": False,
               "finished_at": datetime.now(timezone.utc).isoformat()}
    (output / "export_receipt.json").write_text(json.dumps(receipt, indent=2))
    (output / "neural_models.zip.sha256").write_text(f"{archive_hash}  neural_models.zip\n")
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_models(args.task_root, args.output)


if __name__ == "__main__":
    main()