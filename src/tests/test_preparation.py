import hashlib
import json
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path

from lm.data import Vocab
from lm.preparation import PreparationError, file_sha256, prepare_corpus, repair_variant_overlap, sentences_for, verify_prepared_manifest
from prepare_mega_run import corpora_ready, roc_story_sentences


def make_corpus(directory, count=24):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "tiny.db"
    documents = []
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE Dict (key BLOB PRIMARY KEY, value BLOB)")
        for index in range(count):
            sentences = [f"shared shared unique{index}.", f"ending token{index}."]
            documents.append(sentences)
            value = pickle.dumps([{"sentences": sentences}])
            key = f"/original/corpora/tiny/document{index:03}.txt".encode()
            connection.execute("INSERT INTO Dict VALUES (?, ?)", (key, value))
    checksum = hashlib.sha256()
    checksum.update(database.name.encode() + b"\0" + database.read_bytes())
    Path(str(database) + ".sha").write_text(checksum.hexdigest() + "  tiny.db\n")
    train = directory / "train.txt"
    test = directory / "test.txt"
    train.write_text("\n".join(sentence for document in documents[:-3] for sentence in document) + "\n")
    test.write_text("\n".join(sentence for document in documents[-3:] for sentence in document) + "\n")
    return {"database": str(database), "train": str(train), "test": str(test), "grouping": "source_file"}


def make_quarantine_corpus(directory, empty_document=False):
    entry = make_corpus(directory)
    documents = []
    near_words = [f"word{index}" for index in range(40)]
    with sqlite3.connect(entry["database"]) as connection:
        for index, (key, value) in enumerate(connection.execute("SELECT key,value FROM Dict ORDER BY key").fetchall()):
            sentences = pickle.loads(value)[0]["sentences"]
            if empty_document and index == 0:
                sentences = []
            sentences.append("TEST   Match" if index == 23 else "test match")
            if index < 21 and not (empty_document and index == 0):
                sentences.append("Validation \t COMMON")
            if index in (0, 23) and not empty_document:
                words = list(near_words)
                if index == 23:
                    words[-1] = "different"
                sentences.append(" ".join(words))
            documents.append(sentences)
            connection.execute("UPDATE Dict SET value=? WHERE key=?", (pickle.dumps([{"sentences": sentences}]), key))
    database = Path(entry["database"])
    checksum = hashlib.sha256(database.name.encode() + b"\0" + database.read_bytes()).hexdigest()
    Path(str(database) + ".sha").write_text(checksum + "  tiny.db\n")
    for split, selected in (("train", documents[:-3]), ("test", documents[-3:])):
        Path(entry[split]).write_text("\n".join(sentence for document in selected for sentence in document) + "\n")
    return entry


class MegaReadinessTests(unittest.TestCase):
    def test_explicit_acceptance_retains_flagged_corpora(self):
        statuses = {
            "gutenberg": {"status": "prepared", "near_duplicate_pairs": 2},
            "reuters": {"status": "prepared", "near_duplicate_pairs": 6},
        }
        self.assertFalse(corpora_ready(statuses, set(statuses)))
        self.assertTrue(corpora_ready(statuses, set(statuses), accept_near_duplicates=True))

    def test_acceptance_does_not_allow_missing_or_blocked_corpora(self):
        statuses = {"inaugural": {"status": "prepared", "near_duplicate_pairs": 0}}
        required = {"inaugural", "roc-stories"}
        self.assertFalse(corpora_ready(statuses, required, accept_near_duplicates=True))
        statuses["roc-stories"] = {"status": "blocked"}
        self.assertFalse(corpora_ready(statuses, required, accept_near_duplicates=True))
        self.assertFalse(corpora_ready({}, set(), accept_near_duplicates=True))


class ROCStoryInputTests(unittest.TestCase):
    def test_story_sentences_preserve_complete_text(self):
        row = {"prompt": "First sentence.", "continuation": "Second sentence.",
               "text": "First sentence.  Second sentence."}
        sentences = roc_story_sentences(row, lambda text: ["First sentence.", "Second sentence."])
        self.assertEqual(sentences, ["First sentence.", "Second sentence."])

    def test_accepts_empty_auxiliary_continuation_when_full_text_is_intact(self):
        row = {"text": "Complete story.", "prompt": "Complete story.", "continuation": ""}
        self.assertEqual(roc_story_sentences(row, lambda text: [text]), ["Complete story."])

    def test_accepts_sentence_boundary_whitespace_without_changing_characters(self):
        row = {"text": "She said 'Hello!'.", "prompt": "She said 'Hello!'.", "continuation": ""}
        sentences = ["She said 'Hello!", "'."]
        self.assertEqual(roc_story_sentences(row, lambda text: sentences), sentences)

    def test_rejects_missing_fields_and_reordered_text(self):
        with self.assertRaisesRegex(ValueError, "invalid continuation"):
            roc_story_sentences({"text": "First.", "prompt": "First."}, lambda text: [text])
        with self.assertRaisesRegex(ValueError, "differs"):
            roc_story_sentences({"text": "Second. First.", "prompt": "First.", "continuation": "Second."},
                                lambda text: [text])

    def test_rejects_lossy_segmentation(self):
        row = {"text": "First. Second.", "prompt": "First.", "continuation": "Second."}
        with self.assertRaisesRegex(ValueError, "changed the source text"):
            roc_story_sentences(row, lambda text: ["First."])


class PreparationTests(unittest.TestCase):
    def test_quarantine_precedence_provenance_memberships_and_test_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = make_quarantine_corpus(root / "source")
            raw_bytes = {key: Path(entry[key]).read_bytes() for key in ("database", "train", "test")}
            pilot = prepare_corpus(entry, root / "pilot", budgets=[10, 50, 100])
            prepared = prepare_corpus(entry, root / "derived", budgets=[10, 50, 100],
                                      sentence_quarantine="test_then_validation_v1")
            for key in ("train_documents", "valid_documents", "test_documents", "budgets"):
                self.assertEqual(prepared[key], pilot[key])
            self.assertNotEqual(prepared["fingerprint"], pilot["fingerprint"])
            audit = prepared["sentence_quarantine"]
            self.assertEqual(audit["exclusion_count"], 40)
            self.assertEqual(audit["exact_overlap_after"], {"train_test": 0, "valid_test": 0, "train_valid": 0})
            self.assertEqual(audit["before"]["train"]["documents"], audit["after"]["train"]["documents"])
            self.assertEqual(audit["before"]["test"], audit["after"]["test"])
            self.assertEqual(audit["test_sentence_fingerprint_before"], audit["test_sentence_fingerprint_after"])
            self.assertEqual(audit["test_documents"], pilot["test_documents"])
            self.assertTrue(audit["raw_sources_unchanged"])
            self.assertTrue(audit["test_bytes_preserved"])
            self.assertEqual(audit["test_sha256_before"], audit["test_sha256_after"])
            for exclusion in audit["exclusions"]:
                self.assertIn("source_key", exclusion)
                self.assertIn("document", exclusion)
                self.assertIsInstance(exclusion["position"], int)
                protected_split = "test" if exclusion["normalized"] == "test match" else "valid"
                self.assertEqual(exclusion["reason"], f"matches_{protected_split}")
                self.assertTrue(all(match["split"] == protected_split for match in exclusion["protected_occurrences"]))
                self.assertTrue(all(match["normalized"] == exclusion["normalized"] for match in exclusion["protected_occurrences"]))
            for key, original in raw_bytes.items():
                self.assertEqual(Path(entry[key]).read_bytes(), original)
            near = prepared["duplicate_audit"]["sentences"]
            self.assertEqual(near["pair_count"], 1)
            self.assertEqual(near["status"], "review_required")
            self.assertEqual(near["action"], "audit_only_no_removal")
            self.assertGreaterEqual(near["pairs"][0]["jaccard"], .9)
            self.assertNotEqual(near["pairs"][0]["left"]["normalized"], near["pairs"][0]["right"]["normalized"])
            self.assertEqual(near["min_sentence_tokens"], 10)
            self.assertNotIn("sentence_quarantine", pilot)
            self.assertNotIn("sentences", pilot["duplicate_audit"])
            self.assertEqual(prepare_corpus(entry, root / "derived", budgets=[10, 50, 100],
                                           sentence_quarantine="test_then_validation_v1")["fingerprint"],
                             prepared["fingerprint"])

    def test_quarantine_fails_instead_of_dropping_an_empty_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = make_quarantine_corpus(root / "source", empty_document=True)
            with self.assertRaisesRegex(PreparationError, "empty selected .* document"):
                prepare_corpus(entry, root / "derived", budgets=[10, 100],
                               sentence_quarantine="test_then_validation_v1")
            self.assertFalse((root / "derived" / "manifest.json").exists())

    def test_actual_text_and_partition_changes_invalidate_prepared_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = make_corpus(root / "source")
            prepared = prepare_corpus(entry, root / "prepared", budgets=[10, 100])
            self.assertEqual(verify_prepared_manifest(prepared)["fingerprint"], prepared["fingerprint"])
            changed_partitions = dict(prepared)
            changed_partitions["valid_documents"] = prepared["train_documents"][:1]
            with self.assertRaisesRegex(PreparationError, "identity mismatch"):
                verify_prepared_manifest(changed_partitions)
            with sqlite3.connect(prepared["index"]) as connection:
                connection.execute("""UPDATE sentences SET canonical='different prepared text',
                                   tokens='["different", "prepared", "text"]',length=3 WHERE rowid=1""")
            prepared["index_sha256"] = file_sha256(prepared["index"])
            manifest_path = root / "prepared" / "manifest.json"
            manifest_path.write_text(json.dumps(prepared))
            with self.assertRaisesRegex(PreparationError, "identity mismatch"):
                prepare_corpus(entry, root / "prepared", budgets=[10, 100])

    def test_variant_repair_keeps_test_and_removes_every_overlapping_training_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            database = source / "webtext.db"
            train = []
            test = []
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE Dict (key BLOB PRIMARY KEY,value BLOB)")
                for name in ("a", "b", "c", "d", "e", "f"):
                    for variant in ("raw", "cleaned"):
                        sentence = f"shared shared {name} {variant} variant document."
                        key = f"/original/{variant}/{name}.txt".encode()
                        connection.execute("INSERT INTO Dict VALUES (?,?)", (key, pickle.dumps([{"sentences": [sentence]}])))
                        if name in {"a", "b", "c"} or (name in {"d", "e"} and variant == "cleaned"):
                            train.append(sentence)
                        else:
                            test.append(sentence)
            checksum = hashlib.sha256(database.name.encode() + b"\0" + database.read_bytes()).hexdigest()
            Path(str(database) + ".sha").write_text(checksum + "  webtext.db\n")
            (source / "train.txt").write_text("\n".join(train) + "\n")
            (source / "test.txt").write_text("\n".join(test) + "\n")
            entry = {"database": str(database), "train": str(source / "train.txt"), "test": str(source / "test.txt")}
            result = repair_variant_overlap(entry, root / "derived")
            self.assertEqual(Path(result["test"]).read_bytes(), (source / "test.txt").read_bytes())
            derived = Path(result["train"]).read_text()
            self.assertNotIn(" d ", derived)
            self.assertNotIn(" e ", derived)
            self.assertIn(" a raw ", derived)
            self.assertIn(" a cleaned ", derived)
            prepared = prepare_corpus(result, root / "prepared", budgets=[10, 50, 100])
            self.assertEqual(len(prepared["test_documents"]), 3)
            self.assertEqual(len(prepared["valid_documents"]), 1)
            self.assertEqual(len(prepared["budgets"][-1]["documents"]), 2)

    def test_nested_document_budgets_and_training_only_vocab(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = make_corpus(Path(directory) / "source")
            original = Path(entry["test"]).read_bytes()
            prepared = prepare_corpus(entry, Path(directory) / "prepared", budgets=[10, 50, 100])
            previous = set()
            for budget in prepared["budgets"]:
                current = set(budget["documents"])
                self.assertTrue(previous <= current)
                self.assertFalse(current & set(prepared["valid_documents"]))
                self.assertFalse(current & set(prepared["test_documents"]))
                previous = current
                sentences = sentences_for(prepared, "train", budget["percent"])
                vocab = Vocab(sentences, min_freq=2)
                self.assertNotIn("unique23.", vocab)
                self.assertEqual(len(sentences), len(current) * 2)
                loaded = Vocab.from_list(vocab.itos)
                self.assertEqual(vocab.fingerprint, loaded.fingerprint)
            self.assertEqual(Path(entry["test"]).read_bytes(), original)

    def test_cross_split_document_overlap_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = make_corpus(Path(directory) / "source")
            Path(entry["test"]).write_text("shared shared unique0.\nending token0.\n")
            with self.assertRaises(PreparationError):
                prepare_corpus(entry, Path(directory) / "prepared", budgets=[10, 100])

    def test_unmapped_export_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = make_corpus(Path(directory) / "source")
            Path(entry["test"]).write_text("not in the document store\n")
            with self.assertRaises(PreparationError):
                prepare_corpus(entry, Path(directory) / "prepared", budgets=[10, 100])

    def test_changed_source_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = make_corpus(Path(directory) / "source")
            destination = Path(directory) / "prepared"
            prepare_corpus(entry, destination, budgets=[10, 100])
            Path(entry["train"]).write_text("changed source\n")
            with self.assertRaises(PreparationError):
                prepare_corpus(entry, destination, budgets=[10, 100])


if __name__ == "__main__":
    unittest.main()