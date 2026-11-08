# Synthetic Regression Tests

Run from the repository root after installing [src/requirements.txt](../requirements.txt):

```sh
PYTHONPATH=src ~/.venvs/nlp-lm/bin/python -B -m unittest discover -s src/tests -v
```

The tests use temporary synthetic corpora, SQLite stores and tiny CPU models.
They do not require Google credentials, Drive, CUDA, NLTK corpus downloads or
the private imported datasets. They write no production experiment results.

For a focused check:

```sh
PYTHONPATH=src:src/tests ~/.venvs/nlp-lm/bin/python -B -m unittest test_evaluation.ScoringContractTests -v
PYTHONPATH=src:src/tests ~/.venvs/nlp-lm/bin/python -B -m unittest test_experiments.ExperimentTests -v
```

## Coverage

| Test module | Main checks |
| --- | --- |
| `test_evaluation.py` | Sentence/EOS accounting, aligned count/neural target support, padding, OOV, lazy encoding, honest MLE zeros, CLI structured metrics, conditional accuracy, overflow rendering and legacy/context-sweep FFNN checkpoint loading |
| `test_preparation.py` | Nested training-only document budgets/vocabularies, preserved test bytes, source corruption, missing membership, cross-split overlap and approved variant repair |
| `test_experiments.py` | Tiny train/seal/evaluate/report sweep, all five models, report/CSV ordering, checkpoint-preserving resume, portable bundle contents and tamper rejection |
| `test_trainer.py` | Token-weighted validation NLL, strict patience, restored best checkpoint and epoch-ceiling metadata |
| `test_colab.py` | Generated notebook settings/checksum, clean execution state, compilable cells and pilot-overwrite rejection |
| `test_reporting.py` | Sample seed variability, plateau/ceiling counts and best/last-epoch plots, including legacy metadata |
| `test_analysis.py` | Token-weighted loss reconstruction, honest infinite MLE losses, stable event cohorts, deterministic disjoint case selection, all-seed comparisons and original-versus-mapped counts |
| `test_rare_targets.py` | Exhaustive lexical/combination inventories, all repeated contexts, rare/unseen boundaries, unscorable training-only types, mapped versus lexical accuracy, all-seed wins and empty strata |
| `test_model_comparison_resume.py` | Metric-unit validation, suffixed prediction-backup selection, rare-order keys, interrupted-journal cleanup, duplicate-free export merge and full-run resume wiring |

Fixtures use the same `Dict(key,value)` and paragraph/sentence record shapes as
the imported stores. Keep fixtures small and reuse these modules rather than
copying large private inputs into tests.

## Limits

Passing synthetic tests does not certify GPU capacity, Colab/Drive authorization,
cross-environment bitwise reproducibility, corpus licences, exhaustive duplicate
absence, model convergence or the deferred seven-corpus/multi-seed research run.
The completed real pilot has a separate private verification record; do not
present synthetic test metrics as research results.

The Colab notebook is checked structurally and by compiling its code cells during
release review. Its Drive/upload/export cells must be exercised in Colab, not
executed locally as part of this test suite.

For local editor analysis, select the installed project interpreter rather than
an unrelated system Python. `google.colab` imports are expected to remain
unresolved outside Colab; do not install notebook-only authentication helpers or
disable diagnostics to hide that environment boundary.
