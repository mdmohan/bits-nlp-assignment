# Contributions and AI-Use Disclosure

## Authors

Group 1, Problem Statement 1. The group confirmed this three-author roster:

| Author | Student ID |
|---|---|
| Akshaya Kumar Joish | 2025cs05013 |
| Anuj Kumar Srivastava | 2025cs05040 |
| Murali Das Mohan | 2025cs05052 |

**Contribution statement:** everyone contributed equally.

No granular roles or individual allocations were supplied; none is inferred
from Git/tool activity. Affiliations beyond student IDs, individual signatures
and a final all-author sign-off record were not supplied. This does not assert
that every author signed off on every final artifact. Authors remain responsible
for methods, code, numerical claims, interpretation, attribution and submission.

## AI Assistance

GitHub Copilot assisted with code, audits, figures and manuscript/supporting-doc
preparation. This is development/writing support, not an author, independent
scientific validator or source of experimental data. Claims must remain
traceable to code/configurations/retained evidence; AI suggestions do not replace
the authors' responsibility to verify them.

No pretrained language model was used in the reported nine-model experiment.
FFNN/LSTM models were trained from scratch. Copilot assistance is separate from
experimental architectures/predictions; no pretrained-model or external
language-model API baseline is reported. No new synthetic corpus or
model-generated experimental result is claimed.

## External Code and Sources

The project reuses its existing controller/model/evaluator implementations and
open-source dependencies in [requirements](../../src/requirements.txt), including
PyTorch, NLTK, NumPy, pandas and plotting/data tools. Corpus sources, mirror
declarations and rights limits are in [data access](data_access_instructions.md)
and the [dataset card](dataset_card.md). The
[manuscript](../paper/paper_working_draft.md) cites methods and scholarly sources.
This attributes identified tools/methods, not a completed line-by-line borrowed-code
or license audit.

## Group-Confirmed Instructor Disposition

On 7 November 2026, the group reported instructor approval for supplementary
corpora and reporting disclosed retained-overlap flags in the existing study.
No written approval record, instructor name or approval date was supplied;
7 November is the group confirmation date, not an invented original approval date.

This does not establish statistical independence, zero near overlap or universal
rubric compliance solely through disclosure. Gutenberg retains two train-test
near pairs; Reuters retains six cross-split pairs; ROCStories was not near-scanned.
The approved sentence-prefix LSTM and approved main abstract are retained; no
fixed four-word replacement or new experiment is represented. Remaining limits
are in [reproducibility](reproducibility.md).

The 10 MB group ZIP excludes raw datasets, checkpoints and large scoring-event
archives; this declaration is not a promise of complete artifact access or replay.