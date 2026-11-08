import math
import unittest
from collections import Counter

import numpy as np

from analyze_rare_targets import (
    aggregate_rows, build_inventory, event_metrics, paired_results, rarity, type_results,
)
from lm.data import BOS, EOS, PAD, UNK, Vocab
from lm.evaluation import original_counts


class ExhaustiveRarityTests(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocab.from_list([PAD, UNK, BOS, EOS, "common", "rare"])
        self.words = Counter(common=10, rare=5, singleton=1, absent_from_test=2)
        self.grams = Counter({
            (BOS, "common", "rare"): 5, (BOS, BOS, "common"): 6,
        })
        self.events = [
            dict(event_index=0, target="common", context=[BOS, BOS], eos=False),
            dict(event_index=1, target="rare", context=[BOS, "common"], eos=False),
            dict(event_index=2, target="unseen", context=["common", "rare"], eos=False),
            dict(event_index=3, target="rare", context=["rare", "unseen"], eos=False),
            dict(event_index=4, target="singleton", context=[BOS, BOS], eos=False),
            dict(event_index=5, target=EOS, context=[BOS, "singleton"], eos=True),
        ]

    def test_exact_rarity_boundaries(self):
        self.assertEqual(rarity(0), "unseen_0")
        for count in (1, 2, 5):
            self.assertEqual(rarity(count), "rare_1_5")
        self.assertEqual(rarity(6), "common_6_plus")
        with self.assertRaises(ValueError):
            rarity(-1)

    def test_entire_inventory_and_all_occurrences_without_eos_or_sampling(self):
        inventory, combinations, words, grams, groups = build_inventory(
            self.events, self.words, self.grams, self.vocab,
        )
        by_word = {row["target"]: row for row in inventory}
        self.assertEqual(words["rare"], [1, 3])
        self.assertEqual(by_word["rare"]["training_count"], 5)
        self.assertEqual(by_word["unseen"]["rarity"], "unseen_0")
        self.assertEqual(by_word["unseen"]["heldout_occurrences"], 1)
        self.assertEqual(by_word["singleton"]["mapped_target"], UNK)
        self.assertEqual(by_word["singleton"]["rarity"], "rare_1_5")
        self.assertEqual(by_word["absent_from_test"]["evaluation_status"], "no_heldout_context")
        self.assertEqual(by_word["absent_from_test"]["scored_occurrences"], 0)
        self.assertNotIn("common", by_word)
        self.assertEqual(groups["all_word_targets"], [0, 1, 2, 3, 4])
        self.assertEqual(groups["eligible_union"], [1, 2, 3, 4])
        self.assertEqual(groups["word:rare_1_5"], [1, 3, 4])
        self.assertEqual(groups["trigram:rare_1_5"], [1])
        self.assertEqual(groups["trigram:unseen_0"], [2, 3, 4])
        self.assertEqual(sum(r["heldout_occurrences"] for r in combinations), 4)
        self.assertTrue(all(5 not in indices for indices in groups.values()))
        self.assertEqual(sum(len(x) for x in grams.values()), 4)

    def test_original_rarity_is_not_mapped_unk_frequency(self):
        train = [["common", "singleton"], ["common", "different"]]
        raw = original_counts(train)[3]
        inventory, _, _, _, groups = build_inventory(
            self.events, Counter(word for sentence in train for word in sentence), raw, self.vocab,
        )
        self.assertEqual(next(row for row in inventory if row["target"] == "singleton")["training_count"], 1)
        self.assertEqual(next(row for row in inventory if row["target"] == "unseen")["training_count"], 0)
        self.assertEqual(groups["word:rare_1_5:oov"], [4])

    def test_mapped_accuracy_does_not_claim_original_oov_recovery(self):
        logp = np.array([-1., -1.])
        targets = np.array([1, 5])
        predictions = targets.copy()
        known = np.array([False, True])
        metrics = event_metrics(logp, predictions, targets, known, [0, 1], 2)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["lexical_accuracy"], .5)
        zero = event_metrics(np.array([-math.inf, -1.]), predictions, targets, known, [0], 2)
        self.assertEqual(zero["status"], "infinite_zero_probability")
        self.assertIsNone(zero["nll"])

    def test_paired_wins_use_all_seeds_and_failures_distinguish_unk(self):
        targets = np.array([1, 5])
        known = np.array([False, True])
        scores = {
            ("lstm", 42): (np.array([-.1, -3.]), np.array([1, 4])),
            ("lstm", 43): (np.array([-5., -3.]), np.array([1, 4])),
            ("lstm", 44): (np.array([-5., -3.]), np.array([1, 4])),
            ("ffnn", 42): (np.array([-2., -4.]), np.array([1, 4])),
            ("ffnn", 43): (np.array([-2., -4.]), np.array([1, 4])),
            ("ffnn", 44): (np.array([-2., -4.]), np.array([1, 4])),
        }
        row = paired_results(100, {"example": [0, 1]}, scores, targets, known)[0]
        self.assertEqual(row["lstm_probability_wins"], 1)
        self.assertEqual(row["ffnn_probability_wins"], 1)
        self.assertEqual(row["both_mapped_wrong_all_six"], 1)
        self.assertEqual(row["both_lexically_wrong_all_six"], 2)

    def test_type_results_score_every_repeated_occurrence_and_retain_variability(self):
        scores = {
            ("lstm", seed): (np.array([-a, -2., -b]), np.array([5, 1, 5]))
            for seed, a, b in ((42, 1., 3.), (43, 2., 4.), (44, 3., 5.))
        }
        rows = list(type_results(
            100, "word", {"rare": [0, 2], "no_context": []}, Counter(rare=2), Counter(),
            scores, np.array([5, 1, 5]), np.array([True, False, True]), 3,
        ))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["events"], 2)
        self.assertEqual(rows[0]["seeds"], 3)
        self.assertEqual(rows[0]["nll_mean"], 3)
        self.assertEqual(rows[0]["nll_std"], 1)
        self.assertEqual(rows[0]["lexical_accuracy_mean"], 1)

    def test_empty_groups_stay_explicit_in_seed_summaries(self):
        rows = [
            dict(budget_percent=10, model="lstm", seed=s, group="word:rare_1_5",
                 **event_metrics(np.array([-1.]), np.array([1]), np.array([1]),
                                 np.array([False]), [], 1))
            for s in (42, 43, 44)
        ]
        summary = aggregate_rows(rows)[0]
        self.assertEqual(summary["events"], 0)
        self.assertEqual(summary["status"], "empty")
        self.assertIsNone(summary["lexical_accuracy_mean"])
        self.assertIsNone(summary["nll_mean"])


if __name__ == "__main__":
    unittest.main()
