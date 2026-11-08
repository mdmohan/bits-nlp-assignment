import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from lm.experiments import ExperimentSuite
from lm.colab import create_bundle
from test_preparation import make_corpus


class ExperimentTests(unittest.TestCase):
    def test_sweep_seal_evaluation_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = make_corpus(root / "source")
            config = {
                "schema_version": 1, "name": "tiny_pilot", "purpose": "pilot",
                "output_root": str(root / "results"), "budgets_percent": [10, 100],
                "prepared_data": {"rebuild": False, "sentence_quarantine": "test_then_validation_v1"},
                "vocab": {"mode": "per_budget", "min_freq": 2},
                "corpora": {"tiny": entry}, "device": "cpu",
                "training": {"epochs": 1, "batch_size": 4},
                "models": [
                    {"id": "bigram_mle", "family": "ngram", "order": 2, "smoothing": "mle"},
                    {"id": "trigram_mle", "family": "ngram", "order": 3, "smoothing": "mle"},
                    {"id": "trigram_addk", "family": "ngram", "order": 3, "smoothing": "laplace", "tuning": {"k": [0.1, 1.0]}},
                    {"id": "lstm", "family": "neural", "architecture": {"embed_dim": 4, "hidden_dim": 8, "num_layers": 1, "dropout": 0.0},
                     "tuning": {"lr": [0.001], "seed": 7}, "final_seeds": [42]},
                    {"id": "ffnn", "family": "neural", "architecture": {"embed_dim": 4, "hidden_dim": 8, "context_size": 2, "dropout": 0.0},
                     "tuning": {"lr": [0.001], "seed": 7}, "final_seeds": [42]},
                ],
            }
            path = root / "suite.yaml"
            path.write_text(yaml.safe_dump(config))
            with contextlib.redirect_stdout(io.StringIO()):
                suite = ExperimentSuite(path)
                suite.run("train")
                self.assertFalse(list((root / "results").rglob("test_metrics.json")))
                suite.run("seal", resume=True)
                suite.run("evaluate", resume=True)
                suite.run("report", resume=True)
                self.assertTrue((root / "results" / "report.html").exists())
                self.assertTrue((root / "results" / "summary.csv").exists())
                for filename in ("runs.csv", "metrics.csv", "summary.csv", "diagnostics.csv"):
                    with self.subTest(table=filename):
                        with (root / "results" / filename).open(newline="") as stream:
                            table = list(csv.DictReader(stream))
                        for budget in config["budgets_percent"]:
                            models = dict.fromkeys(row["model"] for row in table if float(row["budget_percent"]) == budget)
                            self.assertEqual(list(models), [spec["id"] for spec in config["models"]])
                self.assertTrue(list((root / "results").rglob("training.png")))
                self.assertTrue(list((root / "results").rglob("perplexity.png")))
                checkpoints = list((root / "results").rglob("*_seed42.pt"))
                before = {str(checkpoint): checkpoint.stat().st_mtime_ns for checkpoint in checkpoints}
                suite.run("train", resume=True)
                after = {str(checkpoint): checkpoint.stat().st_mtime_ns for checkpoint in checkpoints}
            self.assertEqual(before, after)
            self.assertEqual(len(checkpoints), 4)
            metrics = list((root / "results").rglob("test_metrics.json"))
            self.assertEqual(len(metrics), 10)
            for metric in metrics:
                result = json.loads(metric.read_text())
                self.assertEqual(result["tokens"], 21)
                self.assertEqual(result["word_tokens"], 15)
            self.assertTrue((root / "results" / "seal.json").exists())
            for result_path in (root / "results").glob("tiny/budget_*/final/*/seed*/result.json"):
                record = json.loads(result_path.read_text())
                self.assertGreater(record["train_unk_tokens"], 0)
                self.assertEqual(record["train_unk_tokens"], record["train_unk_target_events"])
                self.assertEqual(record["train_unk_tokens"], record["train_unk_input_tokens"])
                self.assertAlmostEqual(record["train_unk_rate"], record["train_unk_tokens"] / record["train_tokens"])
                self.assertAlmostEqual(record["valid_oov_rate"], record["valid_oov_tokens"] / record["valid_word_tokens"])
                if record["family"] == "neural":
                    self.assertEqual(record["best_epoch"], 1)
                    self.assertEqual(record["last_epoch"], 1)
                    self.assertEqual(record["epochs_trained"], 1)
                    self.assertEqual(record["stop_reason"], "max_epochs")
                    self.assertEqual(record["convergence_status"], "budget_limited")
                    self.assertTrue(math.isfinite(record["best_valid_nll"]))
                    self.assertAlmostEqual(record["valid_nll"], record["best_valid_nll"], places=5)
                    self.assertEqual(record["training_target_events_processed"], record["train_prediction_events"])
                    self.assertEqual(record["training_word_tokens_processed"], record["train_tokens"])
                    with (result_path.parent / "history.csv").open(newline="") as stream:
                        history = list(csv.DictReader(stream))
                    self.assertIn("train_nll", history[0])
                    self.assertIn("valid_nll", history[0])
                    self.assertIn("train_tokens", history[0])
                    self.assertIn("valid_tokens", history[0])
                    self.assertIn("train_seconds", history[0])
                    self.assertIn("valid_seconds", history[0])
                suite.verify_run(result_path.parent)
            with contextlib.redirect_stdout(io.StringIO()):
                bundle = create_bundle(suite, root / "colab.zip")
            with zipfile.ZipFile(bundle) as archive:
                self.assertIn("src/run.py", archive.namelist())
                self.assertIn("input/prepared/tiny/sentences.sqlite", archive.namelist())
                remote = yaml.safe_load(archive.read("input/suite.yaml"))
                self.assertTrue(remote["prepared_data"]["packaged_only"])
                self.assertEqual(list(remote["corpora"]), ["tiny"])
                self.assertFalse(any("source/tiny.db" in name for name in archive.namelist()))
            self.assertEqual(len(list((root / "results").glob("tiny/budget_*/vocab.json"))), 2)
            manifest_path = root / "results" / "prepared" / "tiny" / "manifest.json"
            original_manifest = manifest_path.read_text()
            manifest = json.loads(original_manifest)
            manifest["test_documents"] = manifest["valid_documents"]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                suite.run("evaluate", resume=True)
            manifest_path.write_text(original_manifest)
            one_status = next((root / "results").glob("tiny/budget_*/final/lstm/seed42/status.json"))
            status = json.loads(one_status.read_text())
            checkpoint = one_status.parent / next(name for name in status["artifacts"] if name.endswith(".pt"))
            checkpoint.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                suite.run("train", resume=True)


if __name__ == "__main__":
    unittest.main()