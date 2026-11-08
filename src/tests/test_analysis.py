import gzip
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_comparison_sweep import (
    CORRECT, HISTORY_AVAILABLE, HISTORY_SEEN, OOV, PREDICTED_UNK, TARGET_UNK, TestEvents, accuracy_flags,
    accuracy_population, accuracy_rows, history_membership,
    laplace_attribution,
    count_maxima,
    submission_aggregate,
)
from analyze_inaugural import aggregate, contained, coverage_cohort, group_indices, select_cases, summarize_losses
from lm.cli import CountNGram
from lm.data import BOS, EOS, PAD, UNK, Vocab
from lm.evaluation import original_counts
from lm.statistical.ngram import NgramLM
import pandas as pd


class AnalysisTests(unittest.TestCase):
    def test_loss_decomposition_uses_token_weights_not_mean_perplexities(self):
        logp = np.array([-1., -2., -3., -4.])
        target = np.array([4, 1, 4, 3])
        prediction = np.array([4, 5, 6, 3])
        overall = summarize_losses(logp, prediction, target, [0, 1, 2, 3], 4)
        parts = [summarize_losses(logp, prediction, target, ids, 4) for ids in ([0, 2], [1], [3])]
        self.assertEqual(overall["nll"], 2.5)
        self.assertAlmostEqual(sum(p["nll_contribution"] for p in parts), overall["nll"])
        self.assertAlmostEqual(overall["perplexity"], math.exp(2.5))
        self.assertEqual(overall["accuracy"], .5)
        self.assertNotAlmostEqual(sum(p["perplexity"] for p in parts) / 3, overall["perplexity"])

    def test_mle_zeros_empty_groups_and_invalid_probability_are_explicit(self):
        targets = np.array([4, 4])
        predictions = np.array([-1, 4])
        zero = summarize_losses(np.array([-math.inf, -1.]), predictions, targets, [0, 1], 2)
        self.assertEqual(zero["zero_probability_events"], 1)
        self.assertEqual(zero["status"], "infinite_zero_probability")
        self.assertIsNone(zero["nll"])
        self.assertIsNone(zero["perplexity"])
        self.assertEqual(zero["finite_nll_sum"], 1)
        empty = summarize_losses(np.array([-math.inf, -1.]), predictions, targets, [], 2)
        self.assertEqual(empty["status"], "empty")
        self.assertIsNone(empty["nll"])
        for value in (float("nan"), math.inf, .1):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                summarize_losses(np.array([value]), np.array([4]), np.array([4]), [0], 1)

    def test_stable_cohorts_and_membership_monotonicity(self):
        budgets = [10, 50, 100]
        self.assertEqual(coverage_cohort([True, True, True], budgets), "known_all")
        self.assertEqual(coverage_cohort([False, False, False], budgets), "oov_all")
        self.assertEqual(coverage_cohort([False, True, True], budgets), "enters_at_50")
        self.assertEqual(coverage_cohort([False, False, True], budgets), "enters_at_100")
        with self.assertRaisesRegex(ValueError, "not nested"):
            coverage_cohort([True, False, True], budgets)

    def test_group_counts_keep_original_and_mapped_unk_different(self):
        vocab = Vocab.from_list([PAD, UNK, BOS, EOS, "known"])
        train = [["known", "rare"], ["known", "other"]]
        raw = original_counts(train)
        reference = CountNGram(3, vocab.itos, train, allowed_target_ids=vocab.allowed_target_ids).model
        events = [
            dict(event_index=0, target="novel", context=[BOS, "known"], eos=False,
                 target_cohort="oov_all", sequence_cohort="oov_all"),
            dict(event_index=1, target=EOS, context=["known", "novel"], eos=True,
                 target_cohort="eos", sequence_cohort="eos"),
        ]
        groups = group_indices(events, vocab, raw, reference)
        self.assertEqual(groups["original_trigram:unseen"], [0, 1])
        self.assertEqual(groups["mapped_trigram:seen"], [0, 1])
        self.assertEqual(groups["target:oov"], [0])
        self.assertEqual(groups["target:eos"], [1])
        self.assertEqual(groups["target:in_vocabulary"], [])

    def test_selection_is_deterministic_disjoint_diverse_and_all_seed_based(self):
        vocab = Vocab.from_list([PAD, UNK, BOS, EOS, "a", "b", "c"])
        events = [
            dict(event_index=i, event_id=f"doc:0:{i}", sentence_index=i, target=word,
                 context=[BOS, "a"], eos=False)
            for i, word in enumerate(["b", "novel", "c", "a", "b"])
        ]
        raw = {3: Counter({(BOS, "a", "c"): 1, (BOS, "a", "a"): 1})}
        target = np.array([5, 1, 6, 4, 5])
        correct = target.copy()
        wrong = np.array([6, 4, 4, 5, 6])
        scores = {
            ("lstm", 42): (np.array([-2., -3., -.1, -3., -3.]), correct),
            ("lstm", 43): (np.array([-2., -3., -6., -3., -3.]), wrong),
            ("lstm", 44): (np.array([-2., -3., -6., -3., -3.]), wrong),
            ("ffnn", 42): (np.array([-3., -4., -2., -4., -4.]), wrong),
            ("ffnn", 43): (np.array([-3., -4., -2., -4., -4.]), wrong),
            ("ffnn", 44): (np.array([-3., -4., -2., -4., -4.]), wrong),
        }
        # A shared failure must be wrong even for the otherwise accurate LSTM seed.
        scores[("lstm", 42)][1][3] = 5
        quotas = (("known_word_unseen_combination", 1), ("oov_target", 1),
                  ("ffnn_wins", 1), ("shared_failure", 1))
        first, availability = select_cases(events, vocab, raw, scores, quotas)
        second, _ = select_cases(events, vocab, raw, dict(reversed(list(scores.items()))), quotas)
        self.assertEqual(first, second)
        self.assertEqual(len({i for category, i in first}), 4)
        self.assertIn(("oov_target", 1), first)
        self.assertIn(("ffnn_wins", 2), first)
        self.assertIn(("shared_failure", 3), first)
        self.assertEqual(availability["ffnn_wins"]["qualifying_events"], 1)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select_cases(events, vocab, raw, scores, (("oov_target", 2),))

    def test_aggregate_reports_sample_seed_variation_without_exponentiating_mean(self):
        rows = [
            dict(budget_percent=100, model="lstm", group="overall", events=2,
                 event_fraction=1, status="finite", nll=math.log(ppl),
                 perplexity=ppl, accuracy=.5, nll_contribution=math.log(ppl), zero_probability_events=0)
            for ppl in (10., 20., 30.)
        ]
        result = aggregate(rows)[0]
        self.assertEqual(result["perplexity_mean"], 20)
        self.assertEqual(result["perplexity_std"], 10)
        self.assertNotAlmostEqual(math.exp(result["nll_mean"]), result["perplexity_mean"])
        rows[0]["events"] = 1
        with self.assertRaisesRegex(ValueError, "identical"):
            aggregate(rows)

    def test_default_selection_returns_exactly_100_cases_with_declared_quotas(self):
        vocab = Vocab.from_list([PAD, UNK, BOS, EOS, "x", "a", "b", "c"])
        words = ["a"] * 30 + ["novel"] * 30 + ["b"] * 20 + ["c"] * 20
        events = [
            dict(event_index=i, event_id=f"document:{i}:2", sentence_index=i, target=word,
                 context=[BOS, "x"], eos=False)
            for i, word in enumerate(words)
        ]
        raw = {3: Counter({(BOS, "x", "b"): 1, (BOS, "x", "c"): 1})}
        predictions = np.array([vocab.stoi.get(w, vocab.unk_id) for w in words])
        predictions[80:] = vocab.stoi["x"]
        logp = np.full(100, -3.)
        ffnn_logp = logp.copy()
        ffnn_logp[60:80] = -2.
        scores = {
            (model, seed): (values.copy(), predictions.copy())
            for model, values in (("lstm", logp), ("ffnn", ffnn_logp)) for seed in (42, 43, 44)
        }
        selected, _ = select_cases(events, vocab, raw, scores)
        self.assertEqual(len(selected), 100)
        self.assertEqual(len({events[i]["sentence_index"] for _, i in selected}), 100)
        self.assertEqual(Counter(category for category, _ in selected), {
            "known_word_unseen_combination": 30, "oov_target": 30,
            "ffnn_wins": 20, "shared_failure": 20,
        })

    def test_artifact_paths_cannot_escape_input_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(contained(root, "safe/file"), (root / "safe/file").resolve())
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                contained(root, "../escape")


class AccuracyAnalysisTests(unittest.TestCase):
    def test_submission_aggregate_preserves_infinity_missing_sd_and_shared_cost(self):
        common = dict(corpus="example", budget_percent=100)
        rows = [dict(common, model="lstm", seed=str(seed), test_perplexity=ppl,
                     fit_train_seconds=seconds, in_vocabulary_accuracy=math.nan)
                for seed, ppl, seconds in ((13, 10., 2.), (42, 20., 4.), (2026, 30., 6.))]
        rows.extend(dict(common, model=model, seed="", test_perplexity=math.inf,
                         fit_train_seconds=7., in_vocabulary_accuracy=math.nan)
                    for model in ("bigram_mle", "trigram_mle"))
        result = submission_aggregate(pd.DataFrame(rows),
                                      ["test_perplexity", "fit_train_seconds", "in_vocabulary_accuracy"])
        neural = result.set_index("model").loc["lstm"]
        self.assertEqual(neural["test_perplexity_mean"], 20.)
        self.assertEqual(neural["test_perplexity_sd"], 10.)
        self.assertEqual(neural["fit_train_seconds_mean"], 4.)
        self.assertEqual(neural["fit_train_seconds_sd"], 2.)
        for model in ("bigram_mle", "trigram_mle"):
            statistical = result.set_index("model").loc[model]
            self.assertTrue(math.isinf(statistical["test_perplexity_mean"]))
            self.assertTrue(math.isnan(statistical["test_perplexity_sd"]))
            self.assertEqual(statistical["fit_train_seconds_mean"], 7.)
            self.assertTrue(math.isnan(statistical["fit_train_seconds_sd"]))
            self.assertEqual(statistical["recorded_seeds"], "")
        self.assertTrue(result["in_vocabulary_accuracy_mean"].isna().all())

    def test_count_maxima_breaks_only_maximum_count_ties(self):
        profiles = count_maxima({(4, 1): 1, (4, 5): 3, (5, 4): 2, (5, 1): 2})
        self.assertEqual(profiles[(4,)], (3, 5, 1))
        self.assertEqual(profiles[(5,)], (2, 1, 2))

    def test_laplace_uniform_ties_use_allowed_support_and_select_unk(self):
        vocabulary = [PAD, UNK, BOS, EOS, "a", "b"]
        adapter = CountNGram.__new__(CountNGram)
        adapter.smoothing = "laplace"
        adapter.valid_target_ids = [1, 3, 4, 5]
        adapter._best_followers = {}
        adapter.model = NgramLM(2, len(vocabulary), 2, 3, smoothing="laplace",
                               allowed_target_ids=adapter.valid_target_ids)
        for target in adapter.valid_target_ids:
            probability, prediction = adapter.score((4,), target)
            self.assertEqual(probability, .25)
            self.assertEqual(prediction, 1)
        self.assertEqual(adapter.model.prob((4, 0)), 0)
        self.assertEqual(adapter.model.prob((4, 2)), 0)
        adapter.smoothing = adapter.model.smoothing = "mle"
        self.assertEqual(adapter.score((4,), 5), (0, None))

    def test_laplace_seen_argmax_is_mle_argmax_even_when_unk_is_tied(self):
        adapter = CountNGram.__new__(CountNGram)
        adapter.valid_target_ids = [1, 3, 4, 5]
        adapter._best_followers = {(4,): (2, 1)}
        adapter.model = NgramLM(2, 6, 2, 3, smoothing="laplace", allowed_target_ids=adapter.valid_target_ids)
        adapter.model.ngram_counts[2].update({(4, 1): 2, (4, 5): 2})
        adapter.model.context_totals[2][(4,)] = 4
        adapter.smoothing = "laplace"
        self.assertEqual(adapter.score((4,), 5), (3 / 8, 1))
        adapter.smoothing = adapter.model.smoothing = "mle"
        self.assertEqual(adapter.score((4,), 5), (.5, 1))

    def test_laplace_attribution_separates_baseline_and_unseen_gain(self):
        conditions, contexts = [], []
        for order in ("bigram", "trigram"):
            common = dict(corpus="example", percent=20, overall_words=10, unk_targets_words=6,
                          in_vocabulary_correct=1, in_vocabulary_accuracy=.25)
            conditions.extend([dict(common, model=f"{order}_mle", overall_correct=2),
                               dict(common, model=f"{order}_laplace", overall_correct=6,
                                    overall_accuracy=.6, unk_targets_correct=5)])
            for state, group, words, correct, predicted in (("seen", "overall", 5, 2, 2),
                                                            ("unseen", "overall", 5, 4, 5),
                                                            ("unseen", "oov", 4, 4, 4)):
                contexts.append(dict(corpus="example", percent=20, model=f"{order}_laplace",
                                     history_basis="native", history_state=state, target_group=group,
                                     words=words, correct=correct, predicted_unk=predicted))
        baselines, rows = laplace_attribution(pd.DataFrame(conditions), pd.DataFrame(contexts), {("example", 20): 4})
        self.assertEqual(baselines.iloc[0]["accuracy"], .6)
        row = rows.iloc[0]
        self.assertEqual(row["seen_unk_correct"], 1)
        self.assertEqual(row["uniform_tie_accuracy_gain"], .4)
        self.assertEqual(row["accuracy_abstaining_on_unseen"], .2)
        self.assertEqual(row["accuracy_random_tie_expected"], .325)
        self.assertEqual(row["accuracy_minus_unk_baseline"], 0)

    def test_history_membership_includes_eos_and_never_resumes_a_lost_prefix(self):
        mapping = {PAD: 0, UNK: 1, BOS: 2, EOS: 3, "a": 4, "b": 5, "c": 6}
        masks = history_membership([["a", "b"]], [["c", "a", "b"]], mapping)
        self.assertEqual(masks["one_token"].tolist(), [True, False, True, True])
        self.assertEqual(masks["two_token"].tolist(), [True, False, False, True])
        self.assertEqual(masks["sentence_prefix"].tolist(), [True, False, False, False])
        exact = history_membership([["a", "b"]], [["a", "b"]], mapping)
        self.assertTrue(exact["sentence_prefix"].all())
        mapped = history_membership([["rare"]], [["novel"]], mapping)
        self.assertTrue(mapped["sentence_prefix"].all())

    def test_accuracy_population_excludes_eos_and_preserves_empty_groups(self):
        flags = np.array([CORRECT | PREDICTED_UNK, 0, CORRECT], dtype=np.uint8)
        words = accuracy_population(np.array([True, True, False]), flags)
        self.assertEqual(words["words"], 2)
        self.assertEqual(words["accuracy"], .5)
        self.assertEqual(words["predicted_unk_fraction"], .5)
        empty = accuracy_population(np.zeros(3, dtype=bool), flags)
        self.assertTrue(math.isnan(empty["accuracy"]))

    def test_accuracy_rows_use_each_seed_and_keep_literal_unk_distinct(self):
        events = TestEvents.__new__(TestEvents)
        events.eos = np.array([False, False, True])
        first = bytearray([TARGET_UNK | CORRECT | PREDICTED_UNK, OOV | TARGET_UNK, CORRECT])
        second = bytearray([TARGET_UNK, OOV | TARGET_UNK | CORRECT | PREDICTED_UNK, 0])
        events.conditions = {(20, "lstm", "13"): (None, first), (20, "lstm", "42"): (None, second)}
        masks = {name: np.ones(3, dtype=bool) for name in ("one_token", "two_token", "sentence_prefix")}
        rows, contexts = accuracy_rows(events, "example", {20: masks})
        self.assertEqual(rows[0]["in_vocabulary_accuracy"], 1)
        self.assertEqual(rows[1]["in_vocabulary_accuracy"], 0)
        self.assertEqual(rows[0]["literal_unk_words"], 1)
        self.assertTrue(math.isnan(rows[0]["excluding_unk_targets_accuracy"]))
        self.assertEqual(rows[0]["overall_words"], 2)
        self.assertEqual(len(contexts), 36)

    def test_accuracy_flags_keep_target_and_prediction_groups_distinct(self):
        event = dict(target_id=1, prediction_id=1, correct=True, eos=False, oov=True,
                     mapped_target=UNK, context_covered=False)
        flags = accuracy_flags(event)
        self.assertTrue(flags & CORRECT)
        self.assertTrue(flags & TARGET_UNK)
        self.assertTrue(flags & PREDICTED_UNK)
        self.assertTrue(flags & HISTORY_AVAILABLE)
        self.assertFalse(flags & HISTORY_SEEN)
        event.update(target_id=4, correct=False, oov=False, mapped_target="known", context_covered=None)
        flags = accuracy_flags(event)
        self.assertFalse(flags & TARGET_UNK)
        self.assertTrue(flags & PREDICTED_UNK)
        self.assertFalse(flags & HISTORY_AVAILABLE)

    def test_accuracy_flags_reject_inconsistent_correctness_and_eos(self):
        event = dict(target_id=4, prediction_id=5, correct=True, eos=False, oov=False,
                     mapped_target="known", context_covered=True)
        with self.assertRaisesRegex(ValueError, "correctness"):
            accuracy_flags(event)
        event.update(correct=False, eos=True, oov=True)
        with self.assertRaisesRegex(ValueError, "EOS"):
            accuracy_flags(event)

    def test_accuracy_reader_checks_occurrences_not_only_lengths(self):
        first = dict(source="test", percent=20, model="bigram_mle", seed="", sentence_index=0,
                     position=0, raw_target="novel", target_id=1, prediction_id=1, correct=True,
                     mapped_target=UNK, eos=False, oov=True, context_covered=False,
                     log_probability=-1., zero_probability=False)
        second = dict(first, model="lstm", seed=13, context_covered=None)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "events.jsonl.gz"
            with gzip.open(archive, "wt", encoding="utf-8") as handle:
                for event in (first, second):
                    handle.write(json.dumps(event) + "\n")
            events = TestEvents(archive, accuracy=True)
            self.assertTrue(events.flag(20, "lstm", CORRECT)[0])
            self.assertTrue(events.flag(20, "bigram_mle", OOV)[0])
            second["raw_target"] = "different"
            with gzip.open(archive, "wt", encoding="utf-8") as handle:
                for event in (first, second):
                    handle.write(json.dumps(event) + "\n")
            with self.assertRaisesRegex(ValueError, "alignment"):
                TestEvents(archive, accuracy=True)


if __name__ == "__main__":
    unittest.main()
