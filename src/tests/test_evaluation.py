import math
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from lm.cli import CountNGram, LanguageModelCLI
from lm.neural import build_model
from lm.neural.trainer import TrainConfig, Trainer
from lm.statistical.ngram import NgramLM
from lm.evaluation import evaluate_model
from lm.data import SentenceDataset, Vocab


class UniformModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(size))

    def forward(self, inputs):
        return self.logits.expand(*inputs.shape, -1)


class ScoringContractTests(unittest.TestCase):
    def test_cli_returns_shared_structured_metrics(self):
        vocab = Vocab.from_list(["<pad>", "<unk>", "<s>", "</s>", "hello"])
        model = CountNGram(2, vocab.itos, [["hello"]], allowed_target_ids=vocab.allowed_target_ids)
        info = dict(name="bigram", order=2, smoothing="mle", k=1, lm=model,
                    vocab=vocab.itos, stoi=vocab.stoi, lowercase=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            path.write_text("hello\n")
            with contextlib.redirect_stdout(io.StringIO()):
                result = LanguageModelCLI("cpu").evaluate(path, info, None, limit=1, progress_seconds=0)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["ngram"]["tokens"], 2)
        self.assertEqual(result["ngram"]["perplexity"], 1)

    def test_cli_seen_context_accuracy_excludes_unseen_context_predictions(self):
        vocab = Vocab.from_list(["<pad>", "<unk>", "<s>", "</s>", "hello"])
        model = CountNGram(2, vocab.itos, [["hello"]], smoothing="laplace", k=1,
                           allowed_target_ids=vocab.allowed_target_ids)
        info = dict(name="bigram", order=2, smoothing="laplace", k=1, lm=model,
                    vocab=vocab.itos, stoi=vocab.stoi, lowercase=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            path.write_text("missing missing missing\n")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                result = LanguageModelCLI("cpu").evaluate(path, info, None, limit=1, progress_seconds=0)
        self.assertEqual(result["ngram"]["correct"], 2)
        self.assertEqual(result["ngram"]["correct_when_context_seen"], 0)
        row = next(line for line in captured.getvalue().splitlines() if "Top-1 when context seen" in line)
        self.assertEqual(row.split("|")[2].strip(), "0.00%")

    def test_cli_displays_neural_perplexity_overflow_without_raising(self):
        vocab = Vocab.from_list(["<pad>", "<unk>", "<s>", "</s>", "hello"])
        model = UniformModel(len(vocab))
        with torch.no_grad():
            model.logits[vocab.unk_id] = -2000
        info = dict(name="uniform", checkpoint=Path("uniform.pt"), model=model,
                    vocab=vocab.itos, stoi=vocab.stoi, lowercase=True,
                    allowed_target_ids=vocab.allowed_target_ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            path.write_text("missing\n")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                result = LanguageModelCLI("cpu").evaluate(path, None, info, limit=1, progress_seconds=0)
        self.assertEqual(result["neural"]["status"], "overflow")
        self.assertIn("overflow", captured.getvalue())

    def test_cli_matched_ngram_uses_checkpoint_target_support(self):
        vocab = Vocab.from_list(["<pad>", "<unk>", "<s>", "</s>", "hello"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text("hello\n")
            cli = LanguageModelCLI("cpu")
            cli.neural["checkpoint"] = dict(train_path=path.resolve(), lowercase=True,
                                           train_percent=100, vocab=vocab.itos,
                                           allowed_target_ids=vocab.allowed_target_ids)
            with contextlib.redirect_stdout(io.StringIO()):
                cli.load_ngram("matched", 2, path, smoothing="laplace", k=1)
            self.assertEqual(cli.ngrams["matched"]["lm"].model.target_ids, tuple(vocab.allowed_target_ids))

    def test_cli_loads_legacy_and_context_sweep_neural_checkpoints(self):
        vocab = Vocab.from_list(["<pad>", "<unk>", "<s>", "</s>", "hello"])
        params = dict(context_size=2, embed_dim=2, hidden_dim=4, dropout=0)
        config = dict(models=[dict(name="ffnn", params=params)],
                      allowed_target_ids=vocab.allowed_target_ids)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "config.yaml").write_text(yaml.safe_dump(config))
            (run / "vocab.json").write_text(json.dumps(vocab.itos))
            cli = LanguageModelCLI("cpu")
            for name, context_size in (("ffnn_seed42.pt", 2), ("ffnn_ctx4_seed42.pt", 4)):
                with self.subTest(checkpoint=name):
                    model_params = dict(params, context_size=context_size)
                    model = build_model("ffnn", len(vocab), vocab.pad_id, **model_params)
                    checkpoint = run / name
                    torch.save(model.state_dict(), checkpoint)
                    with contextlib.redirect_stdout(io.StringIO()):
                        key = cli.load_neural(checkpoint)
                    loaded = cli.neural[key]
                    self.assertEqual(loaded["model"].context_size, context_size)
                    self.assertEqual(loaded["allowed_target_ids"], vocab.allowed_target_ids)
                    for parameter, value in model.state_dict().items():
                        torch.testing.assert_close(loaded["model"].state_dict()[parameter], value)

    def test_whole_sentence_evaluator_counts_eos_and_unpads(self):
        vocab = Vocab.from_list(["<pad>", "<unk>", "<s>", "</s>", "hello", "world"])
        result = evaluate_model(UniformModel(6), vocab, [["hello", "world"], ["missing"]],
                                family="neural", device=torch.device("cpu"))
        self.assertEqual(result["tokens"], 5)
        self.assertEqual(result["word_tokens"], 3)
        self.assertEqual(result["oov_tokens"], 1)
        self.assertAlmostEqual(result["perplexity"], 4, places=5)
        self.assertAlmostEqual(result["nll_sum"], 5 * math.log(4), places=5)

    def test_dataset_encodes_on_access(self):
        vocab = Vocab.from_list(["<pad>", "<unk>", "<s>", "</s>", "hello"])
        dataset = SentenceDataset([["hello"], [], ["unknown"]], vocab)
        self.assertEqual(len(dataset), 2)
        self.assertFalse(hasattr(dataset, "items"))
        inputs, targets = dataset[0]
        self.assertEqual(inputs.tolist(), [2, 4])
        self.assertEqual(targets.tolist(), [4, 3])

    def test_prepared_line_is_one_sentence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sentences.txt"
            path.write_text("Hello world. Another sentence.\n\n", encoding="utf-8")
            self.assertEqual(
                list(LanguageModelCLI._read_sentences(path)),
                [["Hello", "world.", "Another", "sentence."]],
            )

    def test_addk_normalizes_on_noncontiguous_target_ids(self):
        targets = [1, 3, 5, 7]
        model = NgramLM(
            order=3, vocab_size=8, bos=2, eos=3, smoothing="laplace",
            k=0.1, allowed_target_ids=targets,
        ).fit([[5, 1, 7]])
        for context in ((2, 2), (5, 1), (7, 7)):
            self.assertAlmostEqual(sum(model.prob(context + (target,)) for target in targets), 1)
            self.assertEqual(model.prob(context + (0,)), 0)
            self.assertEqual(model.prob(context + (2,)), 0)
            self.assertAlmostEqual(model.check_normalisation(context), 1)

    def test_neural_loss_uses_same_support_and_excludes_padding(self):
        trainer = Trainer(
            UniformModel(8), TrainConfig(epochs=1), 0, torch.device("cpu"),
            allowed_target_ids=[1, 3, 5, 7],
        )
        loss, count = trainer._batch_loss(
            torch.tensor([[2, 5, 1, 0]]), torch.tensor([[5, 1, 3, 0]])
        )
        self.assertEqual(count, 3)
        self.assertAlmostEqual(loss.item(), 3 * math.log(4), places=5)

    def test_mle_zero_is_not_clipped(self):
        model = NgramLM(2, 6, 2, 3, allowed_target_ids=[1, 3, 4, 5]).fit([[4, 4]])
        self.assertEqual(model.token_logprobs([5])[-1], -math.inf)


if __name__ == "__main__":
    unittest.main()