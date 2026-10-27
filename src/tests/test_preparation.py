import hashlib
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path

from lm.data import Vocab
from lm.preparation import PreparationError, prepare_corpus, repair_variant_overlap, sentences_for


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


class PreparationTests(unittest.TestCase):
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