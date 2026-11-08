"""Prepare immutable local corpus inputs without fitting language models."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lm.preparation import atomic_json, file_sha256


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src/data-pipeline"
ROC_REPOSITORY = "hamishivi/ROCStories"
ROC_REVISION = "cd2b3893a6ee30afada293d5a8f877c3023530e1"


def canonical(sentence):
    return " ".join(sentence.lower().split())


def corpora_ready(statuses, required_corpora, accept_near_duplicates=False):
    return bool(required_corpora) and all(
        statuses.get(corpus, {}).get("status") == "prepared"
        and (accept_near_duplicates or statuses[corpus].get("near_duplicate_pairs") == 0)
        for corpus in required_corpora
    )


def roc_story_sentences(row, sentence_tokenizer):
    for field in ("text", "prompt", "continuation"):
        if not isinstance(row.get(field), str) or (field == "text" and not row[field].strip()):
            raise ValueError(f"ROCStories has an empty or invalid {field}")
    if canonical(row["text"]) != canonical(row["prompt"] + " " + row["continuation"]):
        raise ValueError("ROCStories text differs from prompt plus continuation")
    sentences = [" ".join(sentence.split()) for sentence in sentence_tokenizer(row["text"])]
    if not sentences or any(not sentence for sentence in sentences):
        raise ValueError("ROCStories segmentation produced an empty story or sentence")
    if "".join("".join(sentences).split()) != "".join(row["text"].split()):
        raise ValueError("ROCStories segmentation changed the source text")
    return sentences


def download_roc_stories(snapshot):
    base = f"https://huggingface.co/datasets/{ROC_REPOSITORY}/resolve/{ROC_REVISION}"
    destination = snapshot / "raw" / "roc-stories"
    destination.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://huggingface.co/api/datasets/{ROC_REPOSITORY}/tree/{ROC_REVISION}/data"
    with urllib.request.urlopen(tree_url, timeout=60) as response:
        tree = json.load(response)
    expected_names = {f"data/{split}-00000-of-00001.parquet" for split in ("train", "validation", "test")}
    entries = {entry["path"]: entry for entry in tree if entry.get("type") == "file"}
    if set(entries) != expected_names:
        raise ValueError("ROCStories source file inventory differs from the pinned release")
    checksums = {}
    for name in sorted(expected_names | {"README.md"}):
        target = destination / Path(name).name
        temporary = target.with_suffix(target.suffix + ".part")
        expected = entries.get(name, {}).get("lfs", {}).get("oid")
        if not target.is_file() or (expected and file_sha256(target) != expected):
            print(f"roc-stories: downloading {name}", flush=True)
            with urllib.request.urlopen(f"{base}/{name}?download=true", timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if expected and file_sha256(temporary) != expected:
                raise ValueError(f"ROCStories upstream SHA-256 mismatch: {name}")
            temporary.replace(target)
        checksums[target.name] = file_sha256(target)
    metadata = {"repository": ROC_REPOSITORY, "revision": ROC_REVISION,
                "source_url": f"https://huggingface.co/datasets/{ROC_REPOSITORY}",
                "dataset_card_license": "cc-by-4.0", "upstream_tree": tree,
                "files": checksums, "accessed_at": datetime.now(timezone.utc).isoformat(),
                "source_note": "Public mirror of the combined ROCStories 2016/2017 releases; see preserved dataset card"}
    atomic_json(destination / "download_manifest.json", metadata)
    archive_path = snapshot / "raw/corpora/roc-stories.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for target in sorted(destination.iterdir()):
            if target.is_file() and target.suffix != ".part":
                archive.write(target, target.name)
    return archive_path, metadata


def load_roc_documents(snapshot, destination, sentence_tokenizer):
    import pyarrow.parquet as parquet
    archive_path, metadata = download_roc_stories(snapshot)
    destination.mkdir(parents=True, exist_ok=True)
    documents = []
    story_groups = {}
    source_counts = {}
    empty_continuations = 0
    expected_counts = {"train": 70676, "validation": 7852, "test": 19633}
    with zipfile.ZipFile(archive_path) as archive, (destination / "documents.jsonl").open("w", encoding="utf-8") as parsed:
        for split, expected_count in expected_counts.items():
            name = f"{split}-00000-of-00001.parquet"
            table = parquet.read_table(io.BytesIO(archive.read(name)))
            if table.num_rows != expected_count or not {"text", "prompt", "continuation"}.issubset(table.column_names):
                raise ValueError(f"Unexpected ROCStories schema or row count: {name}")
            source_counts[split] = table.num_rows
            for position, row in enumerate(table.to_pylist()):
                sentences = roc_story_sentences(row, sentence_tokenizer)
                empty_continuations += int(not row["continuation"].strip())
                document = f"roc-stories/{split}/{position:06d}"
                content_hash = hashlib.sha256(canonical(row["text"]).encode()).hexdigest()
                documents.append((document, sentences))
                story_groups[document] = content_hash
                parsed.write(json.dumps({"document": document, "source_file": name, "source_row": position,
                                         "content_sha256": content_hash, "sentences": sentences}, ensure_ascii=False) + "\n")
    source = dict(metadata, archive=str(archive_path.relative_to(snapshot)), archive_sha256=file_sha256(archive_path),
                  document_ids={document: document for document, _ in documents}, story_groups=story_groups,
                  source_counts=source_counts, original_story_ids_available=False,
                  empty_auxiliary_continuations=empty_continuations,
                  document_id_policy="pinned_source_split_and_zero_based_row",
                  split_policy="recombine_mirror_splits_then_partition_whole_canonical_story_groups",
                  preprocessing="NLTK sentence segmentation and whitespace normalization; source text preserved")
    return documents, source


def freeze_splits(documents, partitions, destination, corpus, settings, source):
    destination.mkdir(parents=True, exist_ok=True)
    records = dict(documents)
    assigned = dict(zip(("train", "valid", "test"), partitions))
    retained = {}
    higher = set()
    quarantined = 0
    counts = {}
    with (destination / "occurrences.jsonl").open("w", encoding="utf-8") as occurrences, \
         (destination / "quarantine.jsonl").open("w", encoding="utf-8") as quarantine:
        for split in ("test", "valid", "train"):
            observed = set()
            line_number = 0
            words = 0
            longest = 0
            with (destination / f"{split}.txt").open("w", encoding="utf-8") as output:
                for document in sorted(assigned[split]):
                    document_id = source["document_ids"][document]
                    for position, sentence in enumerate(records[document]):
                        normalized = canonical(sentence)
                        record = {"corpus": corpus, "split": split, "document": document_id,
                                  "sentence_position": position, "original_sentence": sentence}
                        if normalized in higher:
                            quarantine.write(json.dumps(record, ensure_ascii=False) + "\n")
                            quarantined += 1
                            continue
                        line_number += 1
                        token_count = len(sentence.split())
                        words += token_count
                        longest = max(longest, token_count)
                        output.write(sentence + "\n")
                        occurrences.write(json.dumps({**record, "split_line": line_number}, ensure_ascii=False) + "\n")
                        observed.add(normalized)
            if not line_number:
                raise ValueError(f"Quarantine left an empty {corpus}/{split} split")
            retained[split] = observed
            counts[split] = {"documents": len(assigned[split]), "sentences": line_number,
                             "word_tokens": words, "longest_sentence_tokens": longest}
            higher.update(observed)
    overlaps = {"train_valid": len(retained["train"] & retained["valid"]),
                "train_test": len(retained["train"] & retained["test"]),
                "valid_test": len(retained["valid"] & retained["test"])}
    if any(overlaps.values()):
        raise ValueError("Exact normalized sentence overlap remains")
    source = dict(source)
    document_ids = source["document_ids"]
    source["document_ids"] = {source["document_ids"][key]: {"source_key": key,
                              "sentences": len(value)} for key, value in documents}
    atomic_json(destination / "source_manifest.json", source)
    atomic_json(destination / "split_documents.json", {
        split: [document_ids[document] for document in sorted(identifiers)]
        for split, identifiers in assigned.items()
    })
    atomic_json(destination / "leakage_audit.json", {
        "policy": "test_then_validation_v1", "exact_normalization": "lowercase_whitespace",
        "overlaps": overlaps, "quarantined_occurrences": quarantined,
        "near_duplicate_status": "pending_colab_audit", "counts": counts,
    })
    names = ["train.txt", "valid.txt", "test.txt", "occurrences.jsonl", "quarantine.jsonl",
             "source_manifest.json", "split_documents.json", "leakage_audit.json"]
    if (destination / "documents.jsonl").is_file():
        names.append("documents.jsonl")
    manifest = {"schema_version": 1, "corpus": corpus, "settings": settings,
                "budget_unit": "sentence_prefix", "counts": counts,
                "files": {name: file_sha256(destination / name) for name in names}}
    atomic_json(destination / "prepared_manifest.json", manifest)
    return manifest


def prepare(snapshot, selected=None, dry_run=False):
    registry = yaml.safe_load((PIPELINE / "data.yaml").read_text())
    mega = yaml.safe_load((ROOT / "src/configs/mega-run.yaml").read_text())
    configs = [ROOT / "src/configs" / name for name in mega["run"]]
    configured = [yaml.safe_load(config.read_text()) for config in configs]
    if selected and set(selected) - {config["data"]["corpus"] for config in configured}:
        raise ValueError("Unknown corpus selection")
    if dry_run:
        for config in configured:
            corpus = config["data"]["corpus"]
            if not selected or corpus in selected:
                print(f"DATA ONLY: {corpus}; seeds={config['seeds']}; split_seed={config['data']['split_seed']}")
        return
    snapshot.mkdir(parents=True, exist_ok=True)
    nltk_root = snapshot / "nltk_data"
    os.environ["NLTK_DATA"] = str(nltk_root)
    import nltk
    nltk.data.path.insert(0, str(nltk_root))
    if not (nltk_root / "tokenizers/punkt_tab").is_dir():
        if not nltk.download("punkt_tab", download_dir=str(nltk_root), quiet=True):
            raise RuntimeError("Could not acquire the sentence segmentation resource")
    sys.path.insert(0, str(PIPELINE))
    from load_processed import partition_document_ids, _sentences_in_record
    from corpus_parser import create_parser
    from data_process import collect_files
    from data_download import local_archive_path
    from run import verify_prepared_inputs
    status_path = snapshot / "preparation_status.json"
    statuses = json.loads(status_path.read_text()) if status_path.is_file() else {}
    for config in configured:
        corpus = config["data"]["corpus"]
        if selected and corpus not in selected:
            continue
        destination = snapshot / "processed" / corpus
        config_path = snapshot / "configs" / f"{corpus}.yaml"
        if (destination / "prepared_manifest.json").is_file() and config_path.is_file():
            verify_prepared_inputs(yaml.safe_load(config_path.read_text()))
            print(f"{corpus}: verified existing immutable snapshot", flush=True)
            continue
        try:
            entry = registry["corpora"][corpus]
            if entry["download"]["type"] == "local":
                if corpus != "roc-stories":
                    archive = local_archive_path(entry["download"]["package"], ROOT)
                    raise ValueError(f"Unsupported local-source corpus: {corpus}; configured archive: {archive}")
                documents, source = load_roc_documents(snapshot, destination, nltk.sent_tokenize)
                config["data"]["clean"] = False
            else:
                logs = snapshot / "logs"
                logs.mkdir(exist_ok=True)
                log_path = logs / f"{corpus}.log"
                print(f"{corpus}: downloading and running data-only pipeline", flush=True)
                command = [sys.executable, str(PIPELINE / "data_pipeline.py"), "--corpus", corpus,
                           "--download", "--in", str(PIPELINE / "data.yaml"),
                           "--download-out", str(snapshot / "raw"), "--out", str(snapshot / "pipeline")]
                if config["data"].get("clean"):
                    command.append("--clean")
                with log_path.open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, env=os.environ.copy())
                parser_name = corpus
                if corpus not in {"brown", "gutenberg", "reuters"}:
                    parser_name = "general"
                corpus_parser = create_parser(parser_name)
                source_folder = snapshot / "raw/corpora" / entry["download"]["package"]
                if config["data"].get("clean"):
                    source_folder = snapshot / "pipeline" / corpus / "cleaned"
                files = collect_files(source_folder, "", parser_name, entry.get("ignore-files", []))
                documents = []
                destination.mkdir(parents=True, exist_ok=True)
                with (destination / "documents.jsonl").open("w", encoding="utf-8") as parsed:
                    for target in files:
                        record = corpus_parser.parse(target)
                        sentences = _sentences_in_record(record)
                        if not sentences:
                            raise ValueError(f"Source document produced no sentences: {target}")
                        key = str(target.resolve())
                        documents.append((key, sentences))
                        parsed.write(json.dumps({"document": target.relative_to(snapshot).as_posix(),
                                                 "paragraphs": record}, ensure_ascii=False) + "\n")
                archive = snapshot / "raw/corpora" / f"{entry['download']['package']}.zip"
                package = nltk.downloader.Downloader().info(entry["download"]["package"])
                source = {"accessed_at": datetime.now(timezone.utc).isoformat(),
                          "archive": str(archive.relative_to(snapshot)), "archive_sha256": file_sha256(archive),
                          "source_url": package.url, "upstream_checksum": package.checksum,
                          "upstream_sha256": getattr(package, "sha256_checksum", None),
                          "document_ids": {key: Path(key).relative_to(snapshot).as_posix() for key, _ in documents}}
            source.update(registry=entry, nltk_version=nltk.__version__)
            groups = {}
            for document, _ in documents:
                group = source.get("story_groups", {}).get(document, document)
                groups.setdefault(group, []).append(document)
            group_partitions = partition_document_ids(sorted(groups), config["data"]["split_ratios"],
                                                      config["data"]["split_seed"])
            partitions = [[document for group in selected_groups for document in groups[group]]
                          for selected_groups in group_partitions]
            settings = {"split_seed": config["data"]["split_seed"], "split_ratios": config["data"]["split_ratios"],
                        "tokenizer": config["data"].get("tokenizer", "space"),
                        "lowercase": config["data"].get("lowercase", True)}
            manifest = freeze_splits(documents, partitions, destination, corpus, settings, source)
            config["data"].update(output=str(snapshot / "processed"),
                                  input=str(snapshot / "raw"), use_cached=True,
                                  prepared_manifest=str(destination / "prepared_manifest.json"))
            config["execution"] = {"colab_only": True}
            config["device"] = "cuda"
            config_path.parent.mkdir(exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            verify_prepared_inputs(config)
            statuses[corpus] = {"status": "prepared", "counts": manifest["counts"],
                                "manifest_sha256": file_sha256(destination / "prepared_manifest.json")}
            print(f"{corpus}: prepared {manifest['counts']}", flush=True)
        except Exception as error:
            statuses[corpus] = {"status": "blocked", "reason": str(error)}
            print(f"{corpus}: BLOCKED: {error}", flush=True)
        atomic_json(status_path, statuses)
    atomic_json(snapshot / "mega-run.yaml", {"run": [str(snapshot / "configs" / f"{config['data']['corpus']}.yaml")
                                                     for config in configured],
                                             "train_percentages": mega["train_percentages"]})
    atomic_json(snapshot / "mega-smoke.yaml", {"run": [str(snapshot / "configs/inaugural.yaml")],
                                               "train_percentages": [20, 40]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpora", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit", action="store_true", help="Complete data-only near-duplicate and rare-sequence audits")
    parser.add_argument("--accept-near-duplicates", action="store_true", help="Retain near-duplicate audits without blocking launch")
    parser.add_argument("--bundle", type=Path, help="Package frozen inputs and the actual working-tree source")
    args = parser.parse_args()
    prepare(args.output.expanduser().resolve(), args.corpora, args.dry_run)
    if args.audit and not args.dry_run:
        audit_snapshot(args.output.expanduser().resolve(), args.corpora, args.accept_near_duplicates)
    if args.bundle and not args.dry_run:
        create_bundle(args.output.expanduser().resolve(), args.bundle.expanduser().resolve(), args.accept_near_duplicates)


def audit_snapshot(snapshot, selected=None, accept_near_duplicates=False):
    from lm.preparation import _sentence_near_duplicate_audit
    from run import verify_prepared_inputs
    statuses = json.loads((snapshot / "preparation_status.json").read_text())
    for corpus, status in statuses.items():
        if selected and corpus not in selected:
            continue
        if status["status"] != "prepared":
            continue
        config_path = snapshot / "configs" / f"{corpus}.yaml"
        config = yaml.safe_load(config_path.read_text())
        manifest = verify_prepared_inputs(config)
        config["target_support"] = "vocab"
        config["output_dir"] = str(snapshot / "runs")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        directory = snapshot / "processed" / corpus
        if accept_near_duplicates:
            audit = {"status": "not_run_user_accepted", "pair_count": None,
                     "policy": "near-duplicate findings do not block training"}
        else:
            print(f"{corpus}: auditing cross-split sentence similarity", flush=True)
            partitions = json.loads((directory / "split_documents.json").read_text())
            with sqlite3.connect(":memory:") as connection:
                connection.executescript("""
                    CREATE TABLE documents (id TEXT PRIMARY KEY, source_key TEXT);
                    CREATE TABLE sentences (document TEXT, position INTEGER, canonical TEXT, length INTEGER);
                """)
                connection.executemany("INSERT INTO documents VALUES (?,?)",
                                       [(document, document) for identifiers in partitions.values() for document in identifiers])
                with (directory / "occurrences.jsonl").open() as stream:
                    for line in stream:
                        occurrence = json.loads(line)
                        sentence = occurrence["original_sentence"]
                        connection.execute("INSERT INTO sentences VALUES (?,?,?,?)",
                                           (occurrence["document"], occurrence["sentence_position"], sentence,
                                            len(sentence.split())))
                audit = _sentence_near_duplicate_audit(connection, partitions)
        atomic_json(directory / "near_duplicate_audit.json", audit)
        leakage = json.loads((directory / "leakage_audit.json").read_text())
        leakage["near_duplicate_status"] = audit["status"]
        atomic_json(directory / "leakage_audit.json", leakage)
        with (snapshot / "logs" / f"{corpus}_rare.log").open("w") as log:
            for order in (2, 3):
                subprocess.run([sys.executable, str(PIPELINE / "analyze_rare_ngrams.py"),
                                "--train", str(directory / "train.txt"), "--test", str(directory / "test.txt"),
                                "--n", str(order), "--token", config["data"].get("tokenizer", "space")],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        names = {path.name for path in directory.glob("rare-*.csv")}
        names.update({"near_duplicate_audit.json", "leakage_audit.json"})
        manifest["files"].update({name: file_sha256(directory / name) for name in sorted(names)})
        atomic_json(directory / "prepared_manifest.json", manifest)
        verify_prepared_inputs(config)
        status["manifest_sha256"] = file_sha256(directory / "prepared_manifest.json")
        status["near_duplicate_pairs"] = audit["pair_count"]
        status["near_duplicate_status"] = audit["status"]
        print(f"{corpus}: {audit['pair_count']} near-duplicate pairs; rare inventories saved", flush=True)
        atomic_json(snapshot / "preparation_status.json", statuses)


def create_bundle(snapshot, destination, accept_near_duplicates=False):
    from run import verify_prepared_inputs
    statuses = json.loads((snapshot / "preparation_status.json").read_text())
    for corpus, status in statuses.items():
        if status["status"] == "prepared":
            config = yaml.safe_load((snapshot / "configs" / f"{corpus}.yaml").read_text())
            verify_prepared_inputs(config)
            if "near_duplicate_pairs" not in status:
                raise ValueError(f"Complete local audits before packaging: {corpus}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    hashes = {}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def add_bytes(name, content):
            archive.writestr(name, content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        def add_file(target, name):
            archive.write(target, name)
            hashes[name] = file_sha256(target)
        for target in sorted((ROOT / "src").rglob("*")):
            if target.is_file() and target.suffix in {".py", ".yaml", ".yml", ".txt"}:
                if "results" not in target.relative_to(ROOT).parts:
                    add_file(target, target.relative_to(ROOT).as_posix())
        add_file(ROOT / "colab/nltk_data_efficiency.ipynb", "colab/nltk_data_efficiency.ipynb")
        for folder in ("processed", "logs"):
            for target in sorted((snapshot / folder).rglob("*")):
                if target.is_file():
                    add_file(target, "snapshot/" + target.relative_to(snapshot).as_posix())
        for target in sorted((snapshot / "raw/corpora").glob("*.zip")):
            add_file(target, "snapshot/" + target.relative_to(snapshot).as_posix())
        for target in sorted((snapshot / "pipeline").rglob("*")):
            if target.is_file() and target.suffix in {".txt", ".json"}:
                add_file(target, "snapshot/" + target.relative_to(snapshot).as_posix())
        add_file(snapshot / "preparation_status.json", "snapshot/preparation_status.json")
        mega = yaml.safe_load((ROOT / "src/configs/mega-run.yaml").read_text())
        for name in mega["run"]:
            corpus = Path(name).stem
            config_path = snapshot / "configs" / name
            if not config_path.is_file():
                config_path = ROOT / "src/configs" / name
            config = yaml.safe_load(config_path.read_text())
            config["data"].update(output="snapshot/processed", input="snapshot/raw",
                                  prepared_manifest=f"snapshot/processed/{corpus}/prepared_manifest.json")
            config["target_support"] = "vocab"
            config["execution"] = {"colab_only": True}
            config["output_dir"] = "outputs/full"
            config["device"] = "cuda"
            add_bytes(f"snapshot/configs/{name}", yaml.safe_dump(config, sort_keys=False).encode())
        add_bytes("snapshot/mega-run.yaml", yaml.safe_dump({"run": [f"snapshot/configs/{name}" for name in mega["run"]],
                                                           "train_percentages": mega["train_percentages"]}).encode())
        add_bytes("snapshot/mega-smoke.yaml", yaml.safe_dump({"run": ["snapshot/configs/inaugural.yaml"],
                                                             "train_percentages": [20, 40]}).encode())
        packages = {distribution.metadata["Name"]: distribution.version
                    for distribution in importlib.metadata.distributions()}
        add_bytes("snapshot/local_environment.json", json.dumps({"python": sys.version, "packages": packages}, indent=2).encode())
        near_duplicate_policy = "block_flagged_pairs"
        if accept_near_duplicates:
            near_duplicate_policy = "accepted_audit_only"
        manifest = {"schema_version": 1, "files": hashes, "corpora": statuses,
                    "budget_unit": "sentence_prefix", "target_support": "excludes_PAD_BOS_includes_UNK_EOS",
                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                    "training_location": "Colab GPU only",
                    "near_duplicate_policy": near_duplicate_policy,
                    "full_sweep_ready": corpora_ready(statuses, {Path(name).stem for name in mega["run"]},
                                                       accept_near_duplicates)}
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    temporary.replace(destination)
    digest = file_sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(digest + "  " + destination.name + "\n")
    print(f"Bundle: {destination} ({destination.stat().st_size / 1024 ** 2:.1f} MiB); SHA256={digest}", flush=True)


if __name__ == "__main__":
    main()