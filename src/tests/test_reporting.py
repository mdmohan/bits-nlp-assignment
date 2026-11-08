import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lm.reporting import _summary, _training_plot


class FollowupReportingTests(unittest.TestCase):
    def test_seed_variability_and_stopping_counts(self):
        common = dict(
            corpus="inaugural", budget_percent=100, model="lstm", family="neural",
            split="test", status="finite", accuracy=0.1, oov_rate=0.2,
            vocab_size=10, parameters=100, train_seconds=1, zero_probability_events=0,
            train_tokens=20, process_peak_rss_mb=10,
        )
        rows = [
            dict(common, perplexity=10, convergence_status="validation_plateau", best_epoch=4, epochs_trained=7),
            dict(common, perplexity=20, convergence_status="budget_limited", best_epoch=20, epochs_trained=20),
            dict(common, perplexity=30, convergence_status="validation_plateau", best_epoch=3, epochs_trained=6),
        ]
        result = _summary(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result["ppl_mean"], 20)
        self.assertEqual(result["ppl_std"], 10)
        self.assertEqual(result["plateau_seeds"], 2)
        self.assertEqual(result["budget_limited_seeds"], 1)

    def test_plot_handles_best_and_last_epoch_and_legacy_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame(dict(epoch=[1, 2, 3], train_ppl=[9, 7, 5], valid_ppl=[10, 8, 9])).to_csv(
                directory / "history.csv", index=False,
            )
            result_path = directory / "result.json"
            result_path.write_text(json.dumps(dict(best_epoch=2, epochs_trained=3)))
            _training_plot(directory)
            self.assertGreater((directory / "training.png").stat().st_size, 0)
            (directory / "training.png").unlink()
            result_path.write_text("{}")
            _training_plot(directory)
            self.assertTrue((directory / "training.png").exists())


if __name__ == "__main__":
    unittest.main()
