# Experiment Configurations

Run commands from the repository root with the environment described in the
[root README](../../README.md).

## Two Schemas

| Workflow | Example | Entry point |
| --- | --- | --- |
| Legacy neural experiment | `brown.yaml`, `gutenberg.yaml`, `inaugural.yaml`, `reuters.yaml` | `python src/run.py src/configs/brown.yaml` |
| Prepared whole-document sweep | `data_efficiency_nltk_pilot.yaml` | `python src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage all --dry-run` |

Legacy configurations use `models: [{name, params}]`, `seeds` and a `data` section
that invokes the corpus orchestrator. That path can download, load or re-split
data according to its settings. See the [pipeline guide](../data-pipeline/README.md).

The suite uses `schema_version: 1`, `models: [{id, family, ...}]`, `corpora`,
`budgets_percent`, `output_root` and explicit `prepared_data.rebuild: false`.
It consumes already imported corpus exports/document stores instead of calling
the legacy download/split path.

## Pilot Configuration

[data_efficiency_nltk_pilot.yaml](data_efficiency_nltk_pilot.yaml) defines:

- Gutenberg, WebText and Inaugural with source-file grouping and explicit WebText
  overlap repair; the supplied test bytes remain unchanged.
- Seeded whole-document budgets 10%, 20%, ..., 100%, with fixed validation/test
  populations and explicit repeated-budget aliases.
- Training-only vocabulary per budget with `min_freq=2` and no vocabulary cap.
- Bigram/trigram MLE, validation-selected add-k trigram, one-layer LSTM and
  context-2 FFNN; three neural epochs and final seed 42.
- Validation-only add-k candidates `[0.01, 0.1, 1.0]` and explicit resource limits.

Paths in suite configurations resolve relative to the repository root, not the
configuration file's directory. Private corpus contents must match the import
manifest and database checksum sidecars. Do not edit that provenance to accept
unknown replacements.

## Inaugural Follow-Up

[data_efficiency_inaugural_convergence.yaml](data_efficiency_inaugural_convergence.yaml)
is a separate, approved follow-up rather than an edit to the pilot:

- Inaugural only, budgets `[10, 50, 100]`, neural final seeds `[42, 43, 44]`.
- Maximum 20 epochs, patience three, strict token-mean validation NLL improvement
  and restoration of the best checkpoint.
- Unchanged pilot architectures, vocabulary, optimizer and batching settings.
- Opt-in derived sentence quarantine; test takes precedence over validation,
  which takes precedence over training. Original imports and pilot stay intact.
- A new experiment name/output root, fresh statistical baselines and an auditable
  record of exclusions, near-duplicate checks and actual processed training cost.

Runs exhausting the epoch allowance before satisfying patience are
budget-limited, not established convergence.
Use the [Colab follow-up workflow](../../colab/README.md#inaugural-validation-stopping-follow-up)
to generate a checksum-pinned notebook and bundle.

## Start A New Experiment

Copy the suite YAML, choose a new `name` and `output_root`, and make the intended
settings changes there. Run `--dry-run`, then `--stage preflight` or `prepare`
before fitting. Budget percentages, vocabulary policy, validation population,
seeds, model architecture and resource limits are part of the run protocol.

Do not combine `--suite` with legacy `--algorithm`, `--seed`, `--context-size`
or nondefault `--train-percent` overrides. Set suite values in YAML instead.
An existing protocol rejects changed configuration or implementation hashes.

Colab packaging rewrites the suite to portable prepared-index paths. Generate a
matching notebook with `lm.colab --notebook`; it verifies epochs, patience,
budgets and seeds against the frozen bundle. Only the durable output location
changes at execution time. Follow the [Colab guide](../../colab/README.md).