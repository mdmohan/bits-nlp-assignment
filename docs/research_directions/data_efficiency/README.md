# Data-Efficiency Experiment Records

These documents describe the research protocol, implementation traceability and
completed three-corpus Colab pilot. They are not a claim that every planned
seven-corpus acceptance gate has passed.

| Document | Purpose |
| --- | --- |
| [Direction 1](../01_data_efficiency.md) | Scientific question, scope and interpretation |
| [plan.md](plan.md) | Protocol, implementation decisions, tasks and remaining research work |
| [checkpoints/tasks-to-impl.yaml](checkpoints/tasks-to-impl.yaml) | Requirement/task/file mapping and partial completion evidence |
| [batch-report.yaml](batch-report.yaml) | Pilot configuration, verification snapshot, test evidence and limitations |

Operational instructions belong in the [Colab guide](../../../colab/README.md),
[model package guide](../../../src/lm/README.md) and [test guide](../../../src/tests/README.md).

## Completed Pilot

Gutenberg, WebText and Inaugural were run at ten whole-document budgets with
bigram MLE, trigram MLE, validation-selected trigram add-k, LSTM and FFNN. The
three-epoch, single-neural-seed pilot completed 150 final fits and 90 tuning fits.
The private local import verified all final validation/test outputs, 60 neural
histories, ZIP CRCs, 2,434 artifact hashes and 32 input-bundle member hashes.

Result paths in the batch report identify the local verification snapshot. Root
`results/`, checkpoints, original corpora and ZIP bundles are intentionally not
versioned, so these paths are not downloadable GitHub artifacts. Restore the
private result folder to inspect the offline report and original execution logs.
Later report presentation changes do not replace the original archived bundle.

## Interpretation And Remaining Work

- Three epochs and one seed do not establish convergence or variability.
- Each budget rebuilds its vocabulary; cross-budget perplexity also measures
  lexical/output-support growth.
- WebText's approved derived repair preserves test bytes and removes overlapping
  training source groups. Only two training groups remain after validation;
  repeated rounded budgets are aliases, not distinct sample sizes.
- Training histories/fitting statistics exist; separate final training-set
  rescoring was not performed.
- Other corpora, broader resource/provenance checks, richer rare-case selection,
  multi-seed searches and research acceptance/licence declarations remain deferred
  or partial as recorded in the plan.

Keep historical pilot evidence separate from newly added regression tests or
future experiments. A code/configuration change requires a new run identity.