import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from lm.data import Vocab, make_loader
from lm.neural.trainer import Trainer, TrainConfig, set_seed


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        logits = torch.stack((self.weight.expand_as(inputs), torch.zeros_like(inputs)), dim=-1)
        return logits


class TrainerTests(unittest.TestCase):
    def trainer(self, epochs=20, patience=3):
        return Trainer(TinyModel(), TrainConfig(epochs=epochs, patience=patience),
                       pad_id=-1, device=torch.device("cpu"))

    def scripted_fit(self, scores, epochs=20, patience=3, validation=True, **fit_options):
        trainer = self.trainer(epochs, patience)

        def train_epoch(*_):
            with torch.no_grad():
                trainer.model.weight.add_(1)
            trainer.last_train_metrics = {"nll": 2.0, "tokens": 7, "word_tokens": 5, "seconds": 0.1}
            return math.exp(2)

        values = iter(scores)

        def evaluate(*_):
            nll = next(values)
            return {"nll": nll, "perplexity": math.exp(nll) if nll < 700 else math.inf,
                    "tokens": 11, "seconds": 0.2}

        with patch.object(trainer, "train_epoch", side_effect=train_epoch), \
                patch.object(trainer, "evaluate", side_effect=evaluate), \
                contextlib.redirect_stdout(io.StringIO()):
            history = trainer.fit([1], [1] if validation else None, **fit_options)
        return trainer, history

    def test_unequal_token_batches_use_global_nll_weighting(self):
        trainer = self.trainer()
        batches = [(torch.tensor([[0]]), torch.tensor([[1]])),
                   (torch.tensor([[0, 0, 0, 0]]), torch.tensor([[1, 1, 1, 1]]))]

        def batch_loss(inputs, targets):
            count = targets.numel()
            return trainer.model.weight * 0 + (1.0 if count == 1 else 12.0), count

        with patch.object(trainer, "_batch_loss", side_effect=batch_loss), \
                contextlib.redirect_stdout(io.StringIO()):
            train_ppl = trainer.train_epoch(batches)
            valid = trainer.evaluate(batches)
        self.assertAlmostEqual(trainer.last_train_metrics["nll"], 13 / 5)
        self.assertAlmostEqual(train_ppl, math.exp(13 / 5))
        self.assertAlmostEqual(valid["nll"], 13 / 5)
        self.assertEqual(valid["tokens"], 5)

    def test_ties_count_toward_patience_and_restore_best_weights(self):
        trainer, history = self.scripted_fit([3, 2, 2, 2, 2, 1])
        self.assertEqual(len(history), 5)
        self.assertEqual(trainer.model.weight.item(), 2)
        summary = trainer.fit_summary
        self.assertEqual(summary["best_epoch"], 2)
        self.assertEqual(summary["last_epoch"], 5)
        self.assertEqual(summary["stop_reason"], "patience")
        self.assertEqual(summary["convergence_status"], "validation_plateau")
        self.assertTrue(summary["plateau_reached"])
        self.assertFalse(summary["budget_limited"])
        self.assertEqual(summary["training_target_events_processed"], 35)
        self.assertEqual(summary["training_word_tokens_processed"], 25)
        self.assertEqual(history[0]["train_tokens"], 7)
        self.assertEqual(history[0]["valid_tokens"], 11)

    def test_epoch_ceiling_is_not_claimed_as_plateau(self):
        trainer, _ = self.scripted_fit([3, 2, 2.5], epochs=3)
        summary = trainer.fit_summary
        self.assertEqual(summary["stop_reason"], "max_epochs")
        self.assertEqual(summary["convergence_status"], "budget_limited")
        self.assertEqual(summary["best_valid_nll"], 2)
        self.assertEqual(summary["last_valid_nll"], 2.5)
        self.assertAlmostEqual(summary["best_valid_ppl"], math.exp(2))
        self.assertAlmostEqual(summary["last_valid_ppl"], math.exp(2.5))
        self.assertEqual(trainer.model.weight.item(), 2)

    def test_overflowed_perplexities_still_select_strict_nll_improvements(self):
        trainer, _ = self.scripted_fit([1000, 900, 950, 950], epochs=4, patience=2)
        self.assertEqual(trainer.fit_summary["best_epoch"], 2)
        self.assertEqual(trainer.fit_summary["best_valid_nll"], 900)
        self.assertEqual(trainer.fit_summary["best_valid_ppl"], math.inf)
        self.assertEqual(trainer.fit_summary["last_valid_nll"], 950)
        self.assertEqual(trainer.model.weight.item(), 2)
        with patch.object(trainer, "_batch_loss", return_value=(torch.tensor(1000.), 1)), \
                contextlib.redirect_stdout(io.StringIO()):
            metrics = trainer.evaluate([(None, None)])
        self.assertEqual(metrics["nll"], 1000)
        self.assertEqual(metrics["perplexity"], math.inf)

    def test_no_validation_keeps_final_weights_and_no_selection_claim(self):
        trainer, history = self.scripted_fit([], epochs=3, validation=False)
        self.assertEqual(trainer.model.weight.item(), 3)
        self.assertEqual(trainer.fit_summary["stop_reason"], "max_epochs")
        self.assertEqual(trainer.fit_summary["convergence_status"], "no_validation")
        self.assertIsNone(trainer.fit_summary["best_epoch"])
        self.assertIsNone(trainer.fit_summary["best_valid_nll"])
        self.assertTrue(all(row["valid_nll"] is None for row in history))

    def test_real_loader_counts_word_and_target_exposure(self):
        sentences = [["a", "a"], ["b"]]
        vocab = Vocab(sentences, min_freq=2)
        loader = make_loader(sentences, vocab, batch_size=1, shuffle=False)
        model = nn.Sequential(nn.Embedding(len(vocab), 3), nn.Linear(3, len(vocab)))
        trainer = Trainer(model, TrainConfig(epochs=2), vocab.pad_id, torch.device("cpu"), vocab.allowed_target_ids)
        with contextlib.redirect_stdout(io.StringIO()):
            history = trainer.fit(loader, loader)
        self.assertEqual([row["train_tokens"] for row in history], [5, 5])
        self.assertEqual([row["train_word_tokens"] for row in history], [3, 3])
        self.assertEqual(trainer.fit_summary["training_target_events_processed"], 10)
        self.assertEqual(trainer.fit_summary["training_word_tokens_processed"], 6)

    def test_interrupted_resume_matches_uninterrupted_training(self):
        sentences = [["a", "a"], ["b"], ["a", "b"]]
        vocab = Vocab(sentences, min_freq=1)
        loader = make_loader(sentences, vocab, batch_size=1, shuffle=True)

        def make_trainer():
            model = nn.Sequential(nn.Embedding(len(vocab), 3), nn.Dropout(0.2),
                                  nn.Linear(3, len(vocab)))
            return Trainer(model, TrainConfig(epochs=4, patience=10), vocab.pad_id,
                           torch.device("cpu"), vocab.allowed_target_ids)

        def interrupt(row, _model):
            if row["epoch"] == 2:
                raise RuntimeError("simulated interruption")

        for validation in (True, False):
            with self.subTest(validation=validation), tempfile.TemporaryDirectory() as temporary, \
                    contextlib.redirect_stdout(io.StringIO()):
                valid_loader = loader if validation else None
                set_seed(13)
                expected = make_trainer()
                expected_history = expected.fit(loader, valid_loader)
                expected_random = torch.rand(3)
                checkpoint = Path(temporary) / "latest.pt"
                metadata = {"signature": "synthetic-resume"}
                set_seed(13)
                interrupted = make_trainer()
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    interrupted.fit(loader, valid_loader, epoch_callback=interrupt,
                                    checkpoint_path=checkpoint, checkpoint_metadata=metadata)
                saved = torch.load(checkpoint, weights_only=False)
                self.assertEqual(saved["completed_epoch"], 1)
                resumed = make_trainer()
                with self.assertRaisesRegex(ValueError, "do not match"):
                    resumed.fit(loader, valid_loader, checkpoint_path=checkpoint,
                                checkpoint_metadata={"signature": "different"}, resume=True)
                history = resumed.fit(loader, valid_loader, checkpoint_path=checkpoint,
                                      checkpoint_metadata=metadata, resume=True)
                for actual, wanted in zip(history, expected_history):
                    for key in ("epoch", "train_nll", "valid_nll", "train_tokens", "train_word_tokens"):
                        self.assertEqual(actual[key], wanted[key])
                self.assertEqual(resumed.fit_summary, expected.fit_summary)
                for actual, wanted in zip(resumed.model.parameters(), expected.model.parameters()):
                    self.assertTrue(torch.equal(actual, wanted))
                self.assertTrue(torch.equal(torch.rand(3), expected_random))

    def test_resume_preserves_nll_selection_when_perplexity_overflows(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "latest.pt"
            self.scripted_fit([1000, 900], epochs=2, patience=2, checkpoint_path=checkpoint)
            trainer, history = self.scripted_fit([950, 950], epochs=4, patience=2,
                                                checkpoint_path=checkpoint, resume=True)
        self.assertEqual(len(history), 4)
        self.assertEqual(trainer.model.weight.item(), 2)
        self.assertEqual(trainer.fit_summary["best_epoch"], 2)
        self.assertEqual(trainer.fit_summary["best_valid_nll"], 900)
        self.assertEqual(trainer.fit_summary["best_valid_ppl"], math.inf)
        self.assertEqual(trainer.fit_summary["last_valid_nll"], 950)
        self.assertEqual(trainer.fit_summary["stop_reason"], "patience")
        self.assertEqual(trainer.fit_summary["training_target_events_processed"], 28)


if __name__ == "__main__":
    unittest.main()
