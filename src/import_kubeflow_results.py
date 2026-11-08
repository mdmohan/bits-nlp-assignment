"""Restore downloaded Kubeflow artifacts into local results without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def file_hash(filename):
    digest = hashlib.sha256()
    with filename.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_target(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    target = root / relative
    target.resolve().relative_to(root.resolve())
    for candidate in (target, *target.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError(f"Refusing symbolic-link destination: {candidate}")
    return target


def verify_record(filename, record):
    if filename.stat().st_size != record["bytes"]:
        raise ValueError(f"Artifact size mismatch: {filename}")
    if file_hash(filename) != record["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {filename}")


def restore_record(record, destination, backup, opener):
    previous = None
    if destination.exists():
        info = destination.stat()
        placeholder = bool(getattr(info, "st_flags", 0) &
                           getattr(stat, "SF_DATALESS", 0x40000000))
        if placeholder:
            raise ValueError(f"Cloud-only artifact must be hydrated or independently "
                             f"preserved before import: {destination}")
        previous = {"bytes": info.st_size, "sha256": None,
                    "dataless_placeholder": placeholder, "inode": info.st_ino}
        previous["sha256"] = file_hash(destination)
        if (previous["bytes"], previous["sha256"]) == (
                record["bytes"], record["sha256"]):
            return {"status": "unchanged", "preserved": None}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".kubeflow.part")
    with temporary.open("xb") as target:
        try:
            with opener() as source:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        verify_record(temporary, record)
        if previous is not None:
            if backup.exists():
                raise ValueError(f"Preservation destination already exists: {backup}")
            backup.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(backup)
            verify_record(backup, previous)
            previous["path"] = str(backup)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": "restored", "preserved": previous}


def import_results(download, results):
    download = download.resolve()
    results = results.resolve()
    manifest = json.loads((download / "neural_models_manifest.json").read_text())
    validation = json.loads((download / "local_validation.json").read_text())
    archive_path = download / "neural_models.zip"
    verify_record(archive_path, validation)
    records = manifest["files"]
    if len({item["path"] for item in records}) != len(records):
        raise ValueError("Duplicate archive manifest paths")
    training = [item for item in records
                if item["path"].startswith("recovered_run/full/runs/")]
    if len(training) != 420 or manifest["neural_conditions"] != 90:
        raise ValueError("Unexpected recovered neural matrix")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    provenance = results / "comparison_attempts" / "kubeflow_recovery_20261103"
    preserved = results / "comparison_attempts" / ("before_local_import_" + stamp)
    receipt = {"complete": False, "source": str(download), "results": str(results),
               "neural_conditions": 90, "training_files_verified": 0,
               "archive_files": [], "metrics": {}, "backup_metadata": [],
               "preserved_directory": str(preserved), "training_started": False,
               "evaluation_started": False,
               "prior_preserved_directories": [str(item.relative_to(results))
                   for item in sorted((results / "comparison_attempts").glob("before_local_import_*"))
                   if item.is_dir()]}
    expected_members = {item["path"] for item in records}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or
                set(names) != expected_members | {"neural_models_manifest.json"}):
            raise ValueError("Archive membership mismatch")
        if json.loads(archive.read("neural_models_manifest.json")) != manifest:
            raise ValueError("Embedded archive manifest mismatch")
        for item in records:
            relative = item["path"]
            if relative.startswith("recovered_run/full/runs/"):
                destination = safe_target(results, relative.removeprefix("recovered_run/"))
            else:
                destination = safe_target(provenance, relative)
            if archive.getinfo(relative).file_size != item["bytes"]:
                raise ValueError(f"Archive member size mismatch: {relative}")
            backup = safe_target(preserved, destination.relative_to(results))
            outcome = restore_record(item, destination, backup,
                                     lambda: archive.open(relative))
            receipt["archive_files"].append({"source_path": relative,
                                             "path": str(destination.relative_to(results)),
                                             **outcome})
            if len(receipt["archive_files"]) % 50 == 0:
                print("LOCAL_ARCHIVE_PROGRESS", len(receipt["archive_files"]),
                      "/", len(records), flush=True)
        for item in records:
            relative = item["path"]
            if relative.startswith("recovered_run/full/runs/"):
                destination = safe_target(results, relative.removeprefix("recovered_run/"))
                receipt["training_files_verified"] += 1
            else:
                destination = safe_target(provenance, relative)
            verify_record(destination, item)
    print("LOCAL_ARCHIVE_FILES_VERIFIED", len(records), flush=True)
    for corpus in ("gutenberg", "inaugural"):
        metrics_manifest = json.loads((download / (corpus + "_metrics_manifest.json")).read_text())
        metrics_records = metrics_manifest["files"]
        relative_root = Path("full/runs") / (corpus + "_neural") / "metrics"
        source_root = safe_target(download / "drive_restore", relative_root)
        destination_root = safe_target(results, relative_root)
        expected = {item["path"] for item in metrics_records}
        if len(expected) != len(metrics_records):
            raise ValueError("Duplicate metrics manifest paths")
        for item in metrics_records:
            Path(item["path"]).relative_to(relative_root)
        moved = False
        if not destination_root.exists():
            source_paths = {(relative_root / item.relative_to(source_root)).as_posix()
                            for item in source_root.rglob("*") if item.is_file()}
            if source_paths != expected:
                raise ValueError(f"Metrics source inventory mismatch: {corpus}")
            for item in metrics_records:
                source = safe_target(download / "drive_restore", item["path"])
                if source.stat().st_size != item["bytes"]:
                    raise ValueError(f"Metrics source size mismatch: {source}")
            if source_root.stat().st_dev != destination_root.parent.stat().st_dev:
                raise ValueError("Metrics move requires the same filesystem")
            source_root.rename(destination_root)
            moved = True
        actual = {(relative_root / item.relative_to(destination_root)).as_posix()
                  for item in destination_root.rglob("*") if item.is_file()}
        if actual != expected:
            raise ValueError(f"Metrics destination inventory mismatch: {corpus}")
        for item in metrics_records:
            verify_record(safe_target(results, item["path"]), item)
            print("LOCAL_METRIC_VERIFIED", item["path"], flush=True)
        receipt["metrics"][corpus] = {
            "files": len(metrics_records),
            "bytes": sum(item["bytes"] for item in metrics_records),
            "sha256_verified": True, "moved_from_downloads": moved,
            "path": relative_root.as_posix()}
    metadata = sorted(download.glob("*.json")) + sorted(download.glob("*.sha256"))
    for source in metadata:
        relative = Path("backup_metadata") / source.name
        destination = safe_target(provenance, relative)
        record = {"bytes": source.stat().st_size, "sha256": file_hash(source)}
        backup = safe_target(preserved, destination.relative_to(results))
        outcome = restore_record(record, destination, backup, lambda: source.open("rb"))
        verify_record(destination, record)
        receipt["backup_metadata"].append({"path": str(destination.relative_to(results)),
                                            **record, **outcome})
    receipt["complete"] = True
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    provenance.mkdir(parents=True, exist_ok=True)
    receipt_path = provenance / ("local_import_" + stamp + ".json")
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print("LOCAL_RESULTS_IMPORT_COMPLETE", json.dumps({
        "receipt": str(receipt_path), "archive_files_verified": len(records),
        "training_files_verified": receipt["training_files_verified"],
        "neural_conditions": receipt["neural_conditions"],
        "metrics": receipt["metrics"],
        "preserved_files": sum(item["preserved"] is not None
                               for item in receipt["archive_files"])}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    import_results(args.download, args.results)


if __name__ == "__main__":
    main()