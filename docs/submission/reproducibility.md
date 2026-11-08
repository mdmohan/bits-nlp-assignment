# Reproducibility and Submission Boundaries

## Study Identity

Final V3 is Group 1, Problem Statement 1: Brown/Gutenberg/Inaugural/Reuters/
ROCStories/WebText, five 20/40/60/80/100% prefixes of sorted TRAIN SENTENCE LINES,
nine word-level models. It is not the document-budget pilot or press-only Brown.
The approved main abstract and sentence-prefix LSTM are retained. Exact inputs,
rights and audit scope: [data access](data_access_instructions.md),
[dataset card](dataset_card.md).

Vocabulary is rebuilt per prefix: Brown `min_freq=2`, other five 1; lowercase
whitespace; PAD0/UNK1/BOS2/EOS3. Predictions exclude PAD/BOS, include UNK/EOS.
PPL includes EOS; word accuracy/OOV exclude EOS. UNK correctness is not lexical
word recovery. Original neural event context coverage is null/unavailable;
later accuracy audits reconstruct native windows and exact prefix membership.

## Frozen Configuration

| Component | Setting |
|---|---|
| Counts | Bigram/trigram MLE, add-one `k=1`, absolute-discount backoff `d=0.75`; shared trigram counting pass per corpus/budget |
| FFNN context 1/context 2 | Embeddings 128, hidden 256, tanh, dropout 0.3; causal one/two-token histories |
| LSTM | Embeddings 128, hidden 256, one recurrent layer; dropout 0.3 on embeddings/outputs; available sentence prefix/reset per sentence, not four-word |
| Optimization | Adam 0.001, gradient clip 1, batches of up to 32 sentences |
| Selection | Maximum 20 epochs, patience 3, strict token-mean validation NLL, restore best weights |
| Neural seeds | Brown/Gutenberg/Inaugural/Reuters/WebText: 13/42/2026; ROCStories: 13 only |
| Document allocation | Brown/Gutenberg 80/10/10; Inaugural/Reuters/ROCStories 70/20/10; WebText 20/40/40 (one/two/two source files); split seed 42 except WebText 13 |

The [frozen Brown config](../../results/comparison_refresh_20261102_v1/configs/brown.yaml)
and [full manifest](../../results/comparison_refresh_20261102_v1/mega-run.yaml)
are retained-workspace authorities. The matrix has 240 neural fits and 180
statistical conditions, 420 test conditions from 30 shared count passes, not
420 independent replicates. Only 20 serialized count archives remain; all ten
Gutenberg/Inaugural archives are absent.

Three-seed tables use sample SD; ROCStories reports a single run without an
invented zero SD. Its reduced seed coverage reflects computationally constrained
retained execution, not three-seed replication or a measured minimum compute
requirement. No new fitting or inference was run to complete these documents.

## Runtime Evidence, Not a Frozen Lock

Original Kubeflow records identify Python 3.11.11, Torch `2.11.0+cu126`,
CUDA 12.6 and NVIDIA A100-SXM4-80GB. Brown saved training signatures identify
Torch `2.11.0+cu130`; its full original Colab environment was not retained.
Do not conflate CUDA wheel suffixes or fabricate one locked environment from
today's machine. NLTK 3.10.3 is recorded for preparation.

Supplemental Inaugural Torch-summary inspection used CPU Torch 2.14.1 and
torchinfo 1.8.0. Its original generator loaded checkpoints and made dummy
forwards; it is not a training lock and is not rerun for these documentation
checks. Summary shapes, memory and operation estimates are not training peaks.

[Requirements](../../src/requirements.txt) are unchanged lower bounds, not an
exact transitive lock. Local fit-free examples use the existing external
`/Users/ajoish/.venvs/nlp-lm/bin/python`; its packages do not certify original
training. A fresh install can resolve different versions. Preserve original
source/config/environment evidence and label any authorized rerun's new runtime
honestly rather than asserting byte-exact replication.

## Completion and Provenance

Final runs are under `results/comparison_sweep_20261102_v3/full/runs/`:
`brown_press_neural`, `gutenberg_neural`, `inaugural_neural`, `reuters_neural`,
`roc-stories`, `webtext-nn`. Brown is full Brown despite its historical name.

The [Gutenberg/Inaugural receipt](../../results/comparison_sweep_20261102_v3/comparison_attempts/kubeflow_recovery_20261103/local_verification.json)
records complete verification of 90 neural conditions on 3 November 2026.
The [remaining-corpora receipt](../../results/comparison_sweep_20261102_v3/comparison_attempts/kubeflow_remaining_20261105/local_verification.json)
records the Reuters/ROCStories/WebText import on 6 November 2026. The completed
final result folders contain 105 neural checkpoints across those corpora: 45 for
Reuters, 15 for ROCStories and 45 for WebText. Import timestamps do not establish
training completion times. Receipt hash checks happened then; no large-artifact
rehashing is performed in this documentation pass. Historical launch/blocker
text in [Colab notes](../../colab/README.md) is superseded for final inventory
by these receipts/current manuscript.

Saved event/metric verification, vocabularies, histories, manifests and exact
training snapshots support analysis. Missing count archives remain a replay
limit despite complete scored metrics. Null neural context coverage is not unseen.

## Fit-Free Checks and Side Effects

Run from the archive's top-level project directory, preserving `src/colab/docs/results`
relative paths. With retained small run metadata available, these read CSV/JSON
or existing summary assets and write a fresh analysis output, without fitting
or loading checkpoints:

```sh
/Users/ajoish/.venvs/nlp-lm/bin/python -B src/analyze_comparison_sweep.py --corpora brown gutenberg inaugural reuters roc-stories webtext --out results/v3_summary_review
/Users/ajoish/.venvs/nlp-lm/bin/python -B src/analyze_comparison_sweep.py --pipeline --out results/v3_pipeline_review
/Users/ajoish/.venvs/nlp-lm/bin/python -B src/analyze_comparison_sweep.py --architectures --out results/v3_architecture_review
PYTHONPATH=src:src/tests /Users/ajoish/.venvs/nlp-lm/bin/python -B -m unittest test_analysis.AccuracyAnalysisTests -v
```

Architecture mode requires existing Inaugural metadata and supplemental
TXT/HTML/PNG assets, not their checkpoint-inspection generator. Use an already
installed environment; do not blindly run unittest discovery, since other tests
fit models. Synthetic fit-free tests do not certify full corpus/hardware replay.

See [source commands](../../src/README.md) for preparation, frozen training,
comparison, text diagnostics and `--events`/`--accuracy`/`--laplace-audit`.
Event audits are fit-free but read omitted large archives and/or text.
Preparation can download/write data; training fits; comparison fits counts and
performs checkpoint inference. None is an automatic submission check. Renders
write output; preserve original evidence and choose fresh directories.

The implemented complete final-figure flag is `--final-figures`, not the earlier
anticipated `--submission-figures`. Names, options and routing were checked
against current source on 7 November 2026, without executing the renderer:

```sh
/Users/ajoish/.venvs/nlp-lm/bin/python -B src/analyze_comparison_sweep.py --final-figures --runs results/comparison_sweep_20261102_v3/full/runs --corpora brown gutenberg inaugural reuters roc-stories webtext --accuracy-input results/paper_accuracy_20261107 --out results/paper_submission_20261107
```

This requires all six corpora, canonical V3 runs and canonical output (arbitrary
fresh outputs are rejected), plus existing accuracy/Laplace/six-corpus tables
and recovery metadata. It writes the complete PNG/PDF set, 13 CSV tables and
`graph_checks.json`; it does not open events/checkpoints/training text. Unlike
ordinary review renders, this fixed-output recipe can rewrite final evidence;
do not rerun as a docs check or overwrite it without authorization. See the
source guide for complete input/output and mode restrictions.

## Deliverable and Remaining Fields

The main agent's planned final paper is `docs/paper/research_paper.pdf`;
source/group ZIPs are produced separately under `submission/`. These are planned
outputs here, not missing-artifact links or claims of export/package completion.
Preserve `src/colab/docs/results` paths and relative figure/document links.
The ZIP maximum is 10 MB. Include source/notebooks, supporting docs, unchanged
requirements and selected small tables/figures. Exclude raw datasets, checkpoints
and large scoring-event archives. The private frozen bundle is not this ZIP.

The lightweight archive permits inspection, not exact training/event/checkpoint
replay. No public full-artifact location, fresh-machine replay, transitive lock,
live URL availability check, final PDF/ZIP size verification or written approval
record is established here. Final all-author sign-off and affiliations were not
supplied. Group-confirmed instructor approval on 7 November 2026 covers
supplementary corpora/disclosed retained-overlap flags, not independence or zero
near overlap. Contributions and AI use: [declaration](contributions_and_ai.md).
