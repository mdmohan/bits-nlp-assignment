"""Read-only imported-corpus intake and reproducible document partitions."""

import hashlib
import io
import json
import math
import pickle
import random
import shutil
import sqlite3
import sys
from array import array
from collections import deque
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path


class PreparationError(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def readonly(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        yield connection
    finally:
        connection.close()


class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise PreparationError(f"Unexpected object in document store: {module}.{name}")


def source_hashes(entry):
    database = entry.get("database")
    if not database:
        raise PreparationError("Document provenance database is missing; sentence sampling is disabled")
    if entry.get("eligibility") in {"needs_repair", "needs_provenance"}:
        raise PreparationError(f"Corpus requires provenance repair: {entry['eligibility']}")
    if entry.get("grouping", "source_file") != "source_file":
        raise PreparationError("Source variants must be resolved before document preparation")
    database = Path(database)
    for suffix in ("-wal", "-journal"):
        companion = Path(str(database) + suffix)
        if companion.exists() and companion.stat().st_size:
            raise PreparationError(f"Database has an unapplied journal: {companion}")
    pipeline = str(Path(__file__).resolve().parents[1] / "data-pipeline")
    sys.path.insert(0, pipeline)
    try:
        from database_integrity import verify_database_sha
        verify_database_sha(database)
    finally:
        sys.path.remove(pipeline)
    paths = {"database": database, "train": Path(entry["train"]), "test": Path(entry["test"])}
    hashes = {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in paths.items()}
    for name, expected in entry.get("expected_hashes", {}).items():
        if hashes[name]["sha256"] != expected:
            raise PreparationError(f"Imported source checksum mismatch: {name}")
    return hashes


def _index_documents(entry, destination, max_sentence_tokens):
    with readonly(entry["database"]) as source, sqlite3.connect(destination) as index:
        columns = {row[1] for row in source.execute("PRAGMA table_info(Dict)")}
        if columns != {"key", "value"}:
            raise PreparationError("Expected a SQLite shelf with Dict(key,value)")
        index.executescript("""
            CREATE TABLE documents (id TEXT PRIMARY KEY, source_key TEXT, signature TEXT, first TEXT, count INTEGER);
            CREATE TABLE sentences (document TEXT, position INTEGER, canonical TEXT, tokens TEXT, length INTEGER);
            CREATE INDEX sentence_document ON sentences(document, position);
            CREATE INDEX document_first ON documents(first);
        """)
        for key, value in source.execute("SELECT key,value FROM Dict ORDER BY key"):
            record = _PrimitiveUnpickler(io.BytesIO(value)).load()
            if not isinstance(record, list):
                raise PreparationError("Document record must be a list of paragraph objects")
            sentences = []
            for paragraph in record:
                if not isinstance(paragraph, dict) or not isinstance(paragraph.get("sentences", []), list):
                    raise PreparationError("Malformed paragraph record")
                for sentence in paragraph.get("sentences", []):
                    if not isinstance(sentence, str):
                        raise PreparationError("Document sentence must be text")
                    words = sentence.split()
                    if not words:
                        continue
                    if len(words) > max_sentence_tokens:
                        raise PreparationError(f"Sentence exceeds configured resource limit of {max_sentence_tokens} tokens")
                    if any(word.lower() in {"<pad>", "<unk>", "<s>", "</s>"} for word in words):
                        raise PreparationError("Source text contains reserved model tokens")
                    sentences.append(" ".join(words))
            if not sentences:
                continue
            signature = fingerprint(sentences)
            if index.execute("SELECT 1 FROM documents WHERE signature=?", (signature,)).fetchone():
                raise PreparationError("Exact duplicate documents require an explicit repair decision")
            source_key = key
            if isinstance(key, bytes):
                source_key = key.decode("utf-8")
            document_id = fingerprint([Path(source_key).name, signature])
            index.execute("INSERT INTO documents VALUES (?,?,?,?,?)", (document_id, source_key, signature, sentences[0], len(sentences)))
            index.executemany("INSERT INTO sentences VALUES (?,?,?,?,?)", (
                (document_id, position, sentence, json.dumps(sentence.split()), len(sentence.split()))
                for position, sentence in enumerate(sentences)
            ))


def _match_export(connection, path):
    documents = []
    pending = deque()
    with Path(path).open(encoding="utf-8") as stream:
        lines = (" ".join(line.split()) for line in stream)
        lines = (line for line in lines if line)
        exhausted = False
        while pending or not exhausted:
            if not pending:
                try:
                    pending.append(next(lines))
                except StopIteration:
                    break
            candidates = connection.execute("SELECT id,count FROM documents WHERE first=?", (pending[0],)).fetchall()
            if not candidates:
                raise PreparationError(f"Export cannot be reconciled to whole source documents: {path}")
            needed = max(count for _, count in candidates)
            while len(pending) < needed and not exhausted:
                try:
                    pending.append(next(lines))
                except StopIteration:
                    exhausted = True
            matched = []
            for document_id, count in candidates:
                expected = [row[0] for row in connection.execute(
                    "SELECT canonical FROM sentences WHERE document=? ORDER BY position", (document_id,)
                )]
                if len(pending) >= count and all(actual == wanted for actual, wanted in zip(pending, expected)):
                    matched.append((document_id, count))
            if len(matched) != 1:
                raise PreparationError(f"Ambiguous or partial document membership in {path}")
            document_id, count = matched[0]
            if document_id in documents:
                raise PreparationError(f"Repeated document in export: {path}")
            documents.append(document_id)
            for _ in range(count):
                pending.popleft()
    return documents


def _near_duplicate_audit(connection, threshold):
    from datasketch import MinHash, MinHashLSH

    search = MinHashLSH(threshold=threshold, num_perm=128)
    signatures = {}
    for (document_id,) in connection.execute("SELECT id FROM documents ORDER BY id"):
        tokens = []
        for (encoded,) in connection.execute("SELECT tokens FROM sentences WHERE document=? ORDER BY position", (document_id,)):
            tokens.extend(word.lower() for word in json.loads(encoded))
        if len(tokens) < 5:
            continue
        sketch = MinHash(num_perm=128, seed=42)
        for offset in range(len(tokens) - 4):
            sketch.update("\0".join(tokens[offset:offset + 5]).encode())
        for candidate in search.query(sketch):
            if sketch.jaccard(signatures[candidate]) >= threshold:
                raise PreparationError("Potential near-duplicate documents require an explicit repair decision")
        search.insert(document_id, sketch)
        signatures[document_id] = sketch
    return {"exact": "passed", "near": "minhash_lsh_screen_passed", "threshold": threshold, "seed": 42, "num_perm": 128}


def repair_variant_overlap(entry, destination):
    """Preserve test bytes and exclude every training source group represented there."""
    original = {key: value for key, value in entry.items() if key not in {"eligibility", "grouping", "repair_policy"}}
    hashes = source_hashes(original)
    destination = Path(destination).resolve()
    if destination in {Path(entry[key]).resolve().parent for key in ("database", "train", "test")}:
        raise PreparationError("Variant repair must not write to an imported source directory")
    audit_path = destination / "repair.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["original_sources"] != hashes:
            raise PreparationError("Variant repair sources changed; use a new output root")
        for name, expected in audit["derived_hashes"].items():
            if file_sha256(destination / name) != expected:
                raise PreparationError("Derived variant-repair checksum mismatch")
        return audit["entry"]
    records = {}
    with readonly(entry["database"]) as connection:
        for key, value in connection.execute("SELECT key,value FROM Dict ORDER BY key"):
            if isinstance(key, bytes):
                key = key.decode()
            record = _PrimitiveUnpickler(io.BytesIO(value)).load()
            sentences = [" ".join(sentence.split()) for paragraph in record
                         for sentence in paragraph.get("sentences", []) if sentence.strip()]
            if sentences:
                records.setdefault(sentences[0], []).append((Path(key).name, key, sentences))
    grouped = {"train": {}, "test": {}}
    matched_records = {"train": [], "test": []}
    for split in grouped:
        with Path(entry[split]).open(encoding="utf-8") as stream:
            lines = [" ".join(line.split()) for line in stream if line.strip()]
        position = 0
        previous_group = None
        while position < len(lines):
            candidates = [(name, key, sentences) for name, key, sentences in records.get(lines[position], [])
                          if lines[position:position + len(sentences)] == sentences]
            if not candidates or len({name for name, _, _ in candidates}) != 1 or len({fingerprint(sentences) for _, _, sentences in candidates}) != 1:
                raise PreparationError(f"Cannot reconcile source variants in {split} at sentence {position}")
            name, _, sentences = candidates[0]
            if split == "test" and name != previous_group and name in grouped[split]:
                raise PreparationError("Test source groups are interleaved; preserving test order needs a different indexed grouping strategy")
            grouped[split].setdefault(name, []).extend(lines[position:position + len(sentences)])
            matched_records[split].append({"source_group": name, "possible_original_keys": sorted(key for _, key, _ in candidates),
                                           "sentences": len(sentences)})
            previous_group = name
            position += len(sentences)
    overlap = sorted(set(grouped["train"]) & set(grouped["test"]))
    excluded = {name: len(grouped["train"].pop(name)) for name in overlap}
    if len(grouped["train"]) < 2:
        raise PreparationError("Not enough independent training source groups remain after preserving test")
    destination.mkdir(parents=True, exist_ok=True)
    database = destination / "grouped.db"
    temporary = destination / "grouped.db.tmp"
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as connection:
        connection.execute("CREATE TABLE Dict (key BLOB PRIMARY KEY,value BLOB)")
        for split in grouped:
            for name, sentences in grouped[split].items():
                connection.execute("INSERT INTO Dict VALUES (?,?)", (name.encode(), pickle.dumps([{"sentences": sentences}], protocol=4)))
    temporary.replace(database)
    pipeline = str(Path(__file__).resolve().parents[1] / "data-pipeline")
    sys.path.insert(0, pipeline)
    try:
        from database_integrity import write_database_sha
        write_database_sha(database)
    finally:
        sys.path.remove(pipeline)
    with (destination / "train.txt").open("w", encoding="utf-8") as stream:
        for sentences in grouped["train"].values():
            for sentence in sentences:
                stream.write(sentence + "\n")
    shutil.copyfile(entry["test"], destination / "test.txt")
    repaired = {"database": str(database), "train": str(destination / "train.txt"),
                "test": str(destination / "test.txt"), "grouping": "source_file", "derivation_manifest": str(audit_path)}
    audit = {"policy": "preserve_test_drop_all_overlapping_training_source_groups", "original_sources": hashes,
             "excluded_training_groups": excluded, "matched_original_records": matched_records,
             "remaining_train_groups": sorted(grouped["train"]), "test_groups": sorted(grouped["test"]),
             "test_bytes_preserved": file_sha256(entry["test"]) == file_sha256(destination / "test.txt"),
             "variants_retained_within_source_group": True,
             "limitation": "Raw/cleaned variants remain within groups; test retains original duplicated/variant weighting. Few groups produce repeated rounded budgets.",
             "derived_hashes": {name: file_sha256(destination / name) for name in ("grouped.db", "grouped.db.sha", "train.txt", "test.txt")},
             "entry": repaired}
    atomic_json(audit_path, audit)
    return repaired


def prepare_corpus(entry, destination, budgets=None, data_seed=42, budget_seed=55,
                   validation_fraction=0.1, lowercase=True, max_sentence_tokens=2048,
                   near_duplicate_threshold=0.9):
    if budgets is None:
        budgets = list(range(10, 101, 10))
    if sorted(set(budgets)) != budgets or any(percent <= 0 or percent > 100 for percent in budgets):
        raise PreparationError("Budgets must be unique increasing percentages in (0,100]")
    if not 0 < validation_fraction < 1:
        raise PreparationError("Validation fraction must be in (0,1)")
    hashes = source_hashes(entry)
    settings = dict(budgets=budgets, data_seed=data_seed, budget_seed=budget_seed,
                    validation_fraction=validation_fraction, lowercase=lowercase,
                    max_sentence_tokens=max_sentence_tokens, near_duplicate_threshold=near_duplicate_threshold)
    identity = fingerprint({"sources": hashes, "settings": settings, "protocol": 1})
    destination = Path(destination)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != identity:
            raise PreparationError("Prepared sources or protocol changed; use a new output root")
        if file_sha256(manifest["index"]) != manifest["index_sha256"]:
            raise PreparationError("Prepared sentence index checksum mismatch")
        return manifest
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "sentences.sqlite.tmp"
    temporary.unlink(missing_ok=True)
    try:
        _index_documents(entry, temporary, max_sentence_tokens)
        with readonly(temporary) as connection:
            train_documents = _match_export(connection, entry["train"])
            test_documents = _match_export(connection, entry["test"])
            if set(train_documents) & set(test_documents):
                raise PreparationError("A source document appears in both train and test")
            total = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
            if len(train_documents) + len(test_documents) != total:
                raise PreparationError("Document store contains unassigned source records")
            if len(train_documents) < 2 or not test_documents:
                raise PreparationError("At least two training documents and one test document are required")
            audit = _near_duplicate_audit(connection, near_duplicate_threshold)
        shuffled = sorted(train_documents)
        random.Random(data_seed).shuffle(shuffled)
        valid_count = min(max(1, round(len(shuffled) * validation_fraction)), len(shuffled) - 1)
        valid_documents = shuffled[:valid_count]
        training_pool = sorted(shuffled[valid_count:])
        random.Random(budget_seed).shuffle(training_pool)
        allocations = []
        first_budget_for_count = {}
        for percent in budgets:
            count = max(1, math.ceil(len(training_pool) * percent / 100))
            allocation = {"percent": percent, "documents": training_pool[:count]}
            if count in first_budget_for_count:
                allocation["alias_of_percent"] = first_budget_for_count[count]
            else:
                first_budget_for_count[count] = percent
            allocations.append(allocation)
        index_path = (destination / "sentences.sqlite").resolve()
        temporary.replace(index_path)
        manifest = {"schema_version": 1, "fingerprint": identity, "sources": hashes,
                    "settings": settings, "index": str(index_path), "index_sha256": file_sha256(index_path),
                    "valid_documents": valid_documents, "test_documents": test_documents,
                    "budgets": allocations, "duplicate_audit": audit}
        atomic_json(manifest_path, manifest)
        return manifest
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class IndexedSentences(Sequence):
    """Keep sentence positions, not token strings or encoded tensors, in memory."""

    def __init__(self, index_path, documents, lowercase):
        self.index_path = index_path
        self.lowercase = lowercase
        self.row_ids = array("q")
        self.lengths = array("I")
        placeholders = ",".join("?" for _ in documents)
        with readonly(index_path) as connection:
            for row_id, length in connection.execute(
                f"SELECT rowid,length FROM sentences WHERE document IN ({placeholders}) ORDER BY document,position", documents
            ):
                self.row_ids.append(row_id)
                self.lengths.append(length + 2)

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("Use document-budget manifests, not sentence slices")
        with readonly(self.index_path) as connection:
            row = connection.execute("SELECT tokens FROM sentences WHERE rowid=?", (self.row_ids[index],)).fetchone()
        words = json.loads(row[0])
        if self.lowercase:
            return [word.lower() for word in words]
        return words


def sentences_for(manifest, split, percent=None):
    if split == "train":
        documents = next(item["documents"] for item in manifest["budgets"] if item["percent"] == percent)
    else:
        documents = manifest[split + "_documents"]
    return IndexedSentences(manifest["index"], documents, manifest["settings"]["lowercase"])