# Language-Model Package

This package supports two workflows: the original neural controller and an opt-in
whole-document data-efficiency suite. Use the [source guide](../README.md) for
commands and the [Colab guide](../../colab/README.md) for GPU execution.

## Modules

| Module | Responsibility |
| --- | --- |
| `data.py` | One prepared line per sentence, vocabulary, lazy encoded dataset, length-aware batching and optional target masking |
| `controller.py` | Legacy config-driven neural training and seed/context sweeps |
| `cli.py` | Interactive checkpoint inspection, prediction, count-model fitting and structured evaluation |
| `statistical/ngram.py` | Shared count tables and MLE/add-k/backoff implementations |
| `neural/` | Registered FFNN/LSTM architectures and the shared trainer |
| `preparation.py` | Read-only imported SQLite shelves, document reconciliation, repair audits, indexed partitions and nested budgets |
| `evaluation.py` | Aligned count/neural next-token scoring, OOV, accuracy, perplexity and diagnostics |
| `experiments.py` | Preflight, fitting, validation selection, verified resume, sealing and held-out evaluation |
| `reporting.py` | Rebuildable CSV/HTML reports and training/learning curves |
| `colab.py` | Private portable bundle with code, selected prepared indexes and checksums |

The legacy controller's sentence-prefix `--train-percent` is not a document-budget
study. Suite model entries use `id`, `family` and `architecture`; controller entries
use `name` and `params`. Do not interchange the two configuration schemas.

## Suite Lifecycle

1. `preflight`: check imported source hashes and eligibility without training.
2. `prepare`: reconcile exports to whole documents, freeze validation/test
   membership, audit duplicates and record seeded nested training budgets.
3. `train`: build each budget's shared vocabulary, fit fresh models, tune on
   validation only, and save checkpoints, histories, logs and artifact hashes.
4. `seal`: require complete eligible budgets and hash their fits, settings,
   vocabularies, selections and prepared manifests.
5. `evaluate`: verify the seal, then score the fixed held-out test population.
6. `report`: build derived reports from saved fit/evaluation artifacts.

`all` runs training through reporting, including preparation. Use `--dry-run`
before actual execution and a new output root whenever code/configuration changes.
Only identical complete fits with verified hashes may resume. An incomplete fit
restarts, preserving previous attempts; optimizer state is not restored.

## Scoring Contract

One nonempty prepared text line is one independent sentence. Its words plus EOS
are prediction targets; BOS and padding are not targets. The suite masks PAD/BOS
from both neural softmax and count-model smoothing support, keeping UNK/EOS.
Legacy callers can retain their original support by leaving the optional support
argument unset.

Perplexity is `exp(total NLL / target count)`, not an average of sentence scores.
Zero-probability MLE events produce an explicit infinite status with JSON `null`
for unavailable numerical perplexity. Overflow and empty populations are separate
statuses. OOV counts exclude EOS. The CLI uses these structured results for its
display, including conditional seen-context accuracy and overflow handling.

The pilot uses MLE and add-k only; it does not newly certify Kneser-Ney behavior.
Case files retain the first ten events, not a category-selected rare-case study.

## Data And Trust Boundaries

`preparation.py` requires SQLite `Dict(key,value)` input with matching checksum
sidecars and provenance. Primitive records are decoded with a restricted unpickler;
original imports remain read-only. WebText repair is derived separately and never
edits held-out bytes. Near-duplicate screening is probabilistic, not an exhaustive
proof of absence.

Prepared sentence indexes are accessed lazily. Vocabulary size, model parameters,
training events, sentence length and padded/logit allocations have explicit
resource guards. A blocked run is recorded, not silently truncated or capped.

Count checkpoints use Python pickle and must come only from trusted experiments.
Neural `.pt` files contain model weights; the associated config and vocabulary are
required to reconstruct a model. Verifying hashes requires no deserialization.

## Reports And Existing Runs

Table/CSV order within each budget is `bigram_mle`, `trigram_mle`, `trigram_addk`,
`lstm`, `ffnn`. Reports preserve nonfinite statuses and rounded budget aliases.
Cross-budget perplexity includes vocabulary/support growth and is not a
fixed-lexicon comparison.

To rebuild only derived reports for an archived run after reporter code changes,
call `build_report` directly from the repository root:

```sh
PYTHONPATH=src ~/.venvs/nlp-lm/bin/python -c "from lm.reporting import build_report; build_report('results/YOUR_RUN')"
```

The normal suite entry point deliberately refuses changed code/config identities.
Do not reseal or retrain an archived run just to change presentation. Rebuilt
reports reflect the current reporter; preserved input bundles retain the original
execution code. Full artifacts are private and ignored by Git.

Loading a suite checkpoint in the CLI does not reconstruct its whole-document
training subset from an original `train.txt` path. Use the recorded partition and
budget manifests for a scientifically matched count/neural comparison; do not
fit a comparison model on the full imported training pool by accident.

Run the [synthetic regression suite](../tests/README.md) before changing any of
these contracts.