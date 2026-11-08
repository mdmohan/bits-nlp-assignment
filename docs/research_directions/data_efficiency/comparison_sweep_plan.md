# Implementation Plan: Reproducible Comparison Sweep

Date: 2026-11-02

Status: Planning complete; implementation, fresh downloads, upload, and training have not started. ROCStories source acquisition remains blocked. This is a new experiment, not a continuation or replacement of a completed pilot.

## Summary

Combine the automation in [the data-efficiency notebook](../../../colab/nltk_data_efficiency.ipynb) with the checkpoint checks, evaluation, and exports in [the comparison notebook](../../../colab/model_comparison.ipynb). Reuse the experiment suite, canonical models, shared evaluator, and offline reporter. Do not copy notebook training or scoring loops into another implementation.

Refresh all six datasets locally through the existing pipeline, freeze a new document-preserving snapshot, prepare fixed evaluation splits and cumulative training budgets, then upload verified prepared inputs and code to Colab. Train independent models for every condition. Preserve all inputs and outputs needed for replay and downstream analysis.

The user's 2026-11-02 request is the specification. Requirements below are local traceability labels. This plan supersedes the older rollout's corpus list, budget list, and multiple-seed policy only for this new experiment. Historical datasets, notebooks, results, and uncommitted changes stay untouched.

## Experiment Contract

| Setting | New sweep |
| --- | --- |
| Corpora | `brown`, `gutenberg`, `inaugural`, `reuters`, `roc-stories`, `webtext` |
| Training budgets | `20, 40, 60, 80, 100` percent of the frozen training-document pool |
| Seed | `42` everywhere: splitting, budget ordering, sampling/dedup randomness, model initialization, loaders, and any trials; no implicit `55`, `7`, or statistical `0` |
| Splits | New versioned document/source-group split, target ratios 80/10/10 train/validation/test; persist actual counts and rounding |
| Vocabulary | One training-only vocabulary per corpus/budget, shared by all nine models; lowercase whitespace tokenization, `min_freq=2`, no vocabulary cap |
| Smoothing | Fixed add-one `k=1.0`, matching comparison notebook defaults; no add-k search in this matrix |
| Backoff | Existing absolute-discount backoff, `discount=0.75`, after correctness repair and validation; do not relabel it standard Good-Turing Katz |
| Neural training | Existing trainer; proposed common defaults: embedding 128, hidden 256, one LSTM layer, dropout 0.3, Adam learning rate 0.001, batch size 32, maximum 20 epochs, patience 3, gradient clip 1.0 |
| Checkpoint selection | Lowest token-mean validation NLL; record best/last epochs and cap versus patience stop; do not call a capped fit converged |
| Independence | Fresh counts/weights for each budget/model; resume only the same verified job, never warm-start a larger budget |
| Test use | Freeze settings and seal all required fits before test scoring; no test-driven tuning |

Neural defaults are an explicit proposal to freeze before the pilot, not recovered facts about an as-yet-unrun experiment. No extra tuning jobs are planned. Architecture or settings changes after pilot inspection require a new protocol/output identity and a clean final matrix.

### Required Models

| Stable model ID | Family | Exact configuration |
| --- | --- | --- |
| `bigram_mle` | ngram | order 2, MLE |
| `bigram_smoothed` | ngram | order 2, add-one |
| `bigram_backoff` | ngram | order 2, absolute-discount backoff |
| `trigram_mle` | ngram | order 3, MLE |
| `trigram_smoothed` | ngram | order 3, add-one |
| `trigram_backoff` | ngram | order 3, absolute-discount backoff |
| `ffnn_ctx1` | neural | registered architecture `ffnn`, context size 1 |
| `ffnn_ctx2` | neural | registered architecture `ffnn`, context size 2 |
| `lstm` | neural | registered architecture `lstm`, full sentence prefix |

Six corpora x five budgets x nine models = **270 scheduled final fits**: 180 statistical and 90 neural. Each has validation and test metrics, giving **540 final split/model records**. Pilot checks and interrupted attempts are counted separately, not silently included in these totals.

Percentages retain the automated notebook's whole-document semantics, not percentages of sentences or words. Use a persisted seed-42 document order and `ceil(percent * train_document_count / 100)`. Report actual document/sentence/word/event counts and token coverage at every point. Tiny corpora, particularly WebText, can have repeated rounded subsets. Run all requested conditions but mark `alias_of_percent`; these are not five independent data quantities. Do not switch to sentence sampling just to create five distinct points.

## Verified Starting Point

| Inspected owner or focused check | Consequence |
| --- | --- |
| [ExperimentSuite validation and fit dispatch](../../../src/lm/experiments.py) accepts only MLE/add-k statistical specs and does not forward backoff discount. | A nine-entry YAML alone is insufficient; extend the owning adapter and its validation. |
| Statistical final paths currently hard-code seed 0; budget/tuning defaults include 55/7. | Make the new single-seed contract explicit and validated without changing historical protocol interpretation. |
| [The comparison notebook](../../../colab/model_comparison.ipynb), Cells 8, 10, 12, 14, 18, and 20, owns checkpoint checks, split audits, baselines, batch exports, and incremental reports. | Preserve those checks and both reporting contracts through shared functions, not manual checkpoint selection. |
| [The shared evaluator](../../../src/lm/evaluation.py) already exposes word-only accuracy, all-event likelihood, optional event sinks, and examples. | Use it for both notebooks and the suite; stream events rather than retain vocabulary-sized outputs. |
| Main comparison export divides `context_covered` by all prediction events; its incremental report divides word-context coverage by word tokens. | Preserve legacy export semantics explicitly and expose canonical word-context coverage with its denominator; never silently mix these fields. |
| [Backoff implementation](../../../src/lm/statistical/ngram.py) on training `[[2,2],[2,3],[2]]`, BOS 0, EOS 1, allowed targets `[1,2,3]`. | Runtime sums were 0.4375 at bigram context `(2,)` and 0.25 at trigram context `(0,2)`; repair the full-support discount case before accepting either backoff baseline. |
| [Downloader](../../../src/data-pipeline/data_download.py) skips an already-extracted NLTK corpus. | Use a fresh versioned raw root and verify upstream package/index identity; a cache hit is not a freshness check. |
| Local inventory on 2026-11-02. | Brown is absent. Existing Gutenberg/Inaugural/Reuters/WebText imports have train/test but no validation file in their canonical directories. They are historical inputs, not freshly rebuilt data. |
| [ROCStories registry entry](../../../src/data-pipeline/data.yaml) requires `data/raw/roc-stories.zip`; that archive is absent. | Existing `data/processed/rocstory` has flattened train/test exports only. Obtain an authorized source archive/revision and recover story IDs; do not infer stories from line groups. |
| [Pipeline output controls](../../../src/data-pipeline/data_pipeline.py) include `--out` and `--download-out`. | Refresh to new paths instead of copying over imported data or historical experiment outputs. |

These observations supersede incompatible subagent suggestions, including claims that the suite already accepts backoff or that a refreshed snapshot is immutable by default. No blanket corpus-readiness or license claim is made.

## Requirements

| ID | Requirement |
| --- | --- |
| REQ-001 | Acquire current source data for exactly the six requested corpora locally through the existing pipelines, preserving old data and source provenance. |
| REQ-002 | Freeze reproducible, nonempty document/source-group splits with leakage audits and no held-out contamination. |
| REQ-003 | Run the five nested training budgets and persist their exact membership, sizes, rounding, and aliases. |
| REQ-004 | Fit exactly the nine requested model variants independently under their declared smoothing/context settings. |
| REQ-005 | Use the single seed value 42 throughout all random operations and recorded run identities. |
| REQ-006 | Preserve comparison checkpoint/preprocessing/vocabulary checks and identical shared scoring semantics. |
| REQ-007 | Produce comparison-compatible tables/examples and the automated suite's learning, diagnostic, cost, and convergence reports. |
| REQ-008 | Preserve every run's actual train/validation/test inputs, vocabulary, code, configuration, environment, checkpoint, and history for replay. |
| REQ-009 | Automate verified Colab upload, execution, interruption handling, and safe same-job resumption. |
| REQ-010 | Export stable, documented, joinable aggregate and full event-level data for later analyses without retraining. |
| REQ-011 | Gate execution on model correctness, data eligibility, resource checks, and a small end-to-end pilot; never silently shrink scope. |
| REQ-012 | Verify the complete 270-fit/540-metric matrix and checksummed archives, then retrieve and independently verify the local research outputs. |

## Technical Context And Constraints

Use the existing Python/PyTorch/NLTK/PyYAML/pandas/Matplotlib stack, [ExperimentSuite](../../../src/lm/experiments.py), [preparation](../../../src/lm/preparation.py), [evaluation](../../../src/lm/evaluation.py), [reporting](../../../src/lm/reporting.py), and [Colab bundle builder](../../../src/lm/colab.py). Use the explicit external experiment interpreter for local probes; record the actual Colab interpreter and installed package versions separately. Do not infer the editor environment from the shell.

No migration constitution, formal upstream architecture bundle, or topology was supplied. This is an extension of the existing experiment plan, not an application migration. Existing ownership boundaries and the user request supply the constraints. The guidelines lookup has no applicable framework migration rules. Python fact-grounded inspection and a focused runtime probability check support the decisions above.

Fail closed for the requested full sweep: a missing/blocked corpus, failed budget, invalid backoff distribution, changed prepared index, or resource violation prevents full-run sign-off. Partial diagnostics may be generated but must say incomplete and list missing conditions. Keep code/config hashing that includes uncommitted source bytes; a Git commit ID alone cannot reproduce this dirty worktree.

## Implementation Steps And Tasks

All tasks are unimplemented. Proposed new file paths below are targets, not claims that those files already exist.

### 1.1 Fresh Local Source Snapshots

Requirements: REQ-001, REQ-008, REQ-011. Inputs: pipeline downloader, registry, corpus parsers, current missing-input inventory.

- [ ] T001 [Plan:1.1] Add an opt-in refresh driver `src/prepare_comparison_sweep.py` and new registry snapshot under `data/raw/comparison_sweep_v1/`; invoke `src/data-pipeline/data_pipeline.py` with isolated raw/processed roots, capture commands/logs, inventory preexisting files, and never write over canonical imports.
- [ ] T002 [Plan:1.1] Refresh Brown/Gutenberg/Inaugural/Reuters/WebText into that empty raw root; retain source archives, upstream index/package checksums/revisions, access times, NLTK version, registry settings, and every parsed/skipped/failed document. Block any unacknowledged parse failure; do not silently use cached inputs.
- [ ] T003 [Plan:1.1] Obtain and verify the authorized ROCStories source archive and release, add story-ID-aware ingestion through `src/data-pipeline/corpus_parser.py` as needed, and retain original row IDs/text. Record license/source terms without asserting a license from an unverified mirror. Preserve the old `rocstory` directory; use `roc-stories` for the new canonical snapshot.

### 1.2 Frozen Partitions And Budget Inputs

Requirements: REQ-002, REQ-003, REQ-005, REQ-008. Inputs: fresh parsed documents, existing document preparation and quarantine policy.

- [ ] T004 [Plan:1.2] Extend `src/lm/preparation.py` to consume fresh trusted document stores and explicit three-way split manifests, with stable corpus-relative document/story IDs and seed 42. Prefer honoring the pipeline's fixed validation file over silently re-deriving a different split; persist actual 80/10/10 rounding with all partitions nonempty.
- [ ] T005 [Plan:1.2] Apply source/variant grouping before splitting, use one canonical WebText representation per source, audit exact normalized sentence overlaps and declared near-duplicate candidates, and apply versioned test-then-validation sentence quarantine. Save removed occurrences and before/after counts; preserve the newly frozen test population and block unresolved substantial leakage.
- [ ] T006 [Plan:1.2] Materialize `train.txt`, `valid.txt`, `test.txt`, stable sentence/occurrence maps, nested budget document lists, per-budget `train.txt`, vocabulary/counts/support hashes, actual size statistics, and alias records under `data/processed/comparison_sweep_v1/` and prepared experiment inputs. Verify nesting, train-only vocabulary construction, held-out event identity, and source-to-export mapping before upload.

### 2.1 Statistical Correctness And Nine-Model Dispatch

Requirements: REQ-004, REQ-005, REQ-006, REQ-011. Inputs: existing model registry, CountNGram adapter, failing backoff probe.

- [ ] T007 [Plan:2.1] Repair `src/lm/statistical/ngram.py` so absolute-discount backoff normalizes over the explicit allowed target set, including contexts whose observed followers exhaust that set; validate discount bounds. Keep MLE zero probabilities legitimate and avoid unrelated KN changes.
- [ ] T008 [Plan:2.1] Extend `src/lm/experiments.py` to accept backoff specs, forward discount, use distinct `ffnn_ctx1`/`ffnn_ctx2` IDs with `model: ffnn`, and record seed 42 for statistical and neural final jobs under the opt-in new protocol. Preserve compatibility with historical runs.
- [ ] T009 [Plan:2.1] Extend existing tests in `src/tests/test_experiments.py` and `src/tests/test_scoring.py` for all nine variants, full-support/seen/unseen backoff normalization, target masks, fresh fits, FFNN context collision rejection, wrong-subset checkpoint rejection, and single-seed identity. Run the exact failing normalization probe again.

### 2.2 Explicit New Protocol And Replay Identity

Requirements: REQ-003, REQ-004, REQ-005, REQ-008, REQ-011. Inputs: prepared manifests and validated adapters.

- [ ] T010 [Plan:2.2] Add `src/configs/data_efficiency_comparison.yaml` with the exact six corpora/five budgets/nine IDs, fixed `k=1`, discount 0.75, `data_seed=budget_seed=42`, final seeds `[42]`, explicit trial/dedup seeds 42 where applicable, no tuning grids, and a fresh results root. Reject hidden conflicting seed defaults.
- [ ] T011 [Plan:2.2] Include actual prepared-index digests, split/budget memberships, vocabulary/support hashes, model/context settings, code bytes, config, preprocessing, seed, and environment in the immutable run identity in `src/lm/experiments.py`. Same-condition resumes verify artifacts; different budget or source bytes must not reuse a checkpoint.
- [ ] T012 [Plan:2.2] Record Python/NumPy/PyTorch/CUDA RNG and deterministic settings, loader/sampler state, optimizer, epoch, best-checkpoint state, and resume policy using `src/lm/neural/trainer.py` and the suite adapter. If exact interrupted continuation is not supported, restart that fit from seed 42 and preserve failed attempts; never call a weights-only continuation an exact replay.

### 3.1 One Evaluation And Compatibility Contract

Requirements: REQ-006, REQ-007, REQ-010. Inputs: comparison notebook Cells 8/10/12/14/18/20 and shared evaluator.

- [ ] T013 [Plan:3.1] Extract shared comparison validation/export helpers into `src/lm/reporting.py` and call them from the suite and `colab/model_comparison.ipynb`. Check exact vocabulary order/hash, tokenizer/lowercase, corpus, source bytes, split identity, budget membership/hash, target support, seed, and model context before any mixed comparison. Replace manual checkpoint discovery with explicit run manifests for the sweep.
- [ ] T014 [Plan:3.1] Reuse `src/lm/evaluation.py` for identical validation/test scores: one EOS per sentence for NLL/PPL, no PAD/BOS targets, word-only top-1 accuracy, mapped-UNK accuracy distinct from lexical recovery, OOV, MLE zero events, explicit overflow/infinite status, and both legacy and word-only context-coverage denominators. Never resample or re-segment prepared lines.
- [ ] T015 [Plan:3.1] Use the evaluator's event sink to stream complete validation/test event data to compressed JSONL or an equivalent structured existing format. Persist stable corpus/document/sentence/position IDs, original/mapped targets, OOV/EOS flags, prediction, log probability, zero flag, current raw/mapped frequency counts, and group labels. Keep bounded examples for display, not as the only reusable evaluation output; test event-derived metrics against aggregates.

### 3.2 Comparison Tables And Research Reports

Requirements: REQ-007, REQ-008, REQ-010. Inputs: saved metrics, event streams, existing offline reporter.

- [ ] T016 [Plan:3.2] Extend `src/lm/reporting.py` to emit each corpus/budget's comparison CSVs, examples, split/vocabulary/OOV audit, incremental word-metric view, and combined schema-versioned tables. Preserve field names/units from both comparison views with documented aliases rather than silently changing old columns.
- [ ] T017 [Plan:3.2] Add explicit nine-model ordering/labels and configured budget ticks, per-corpus PPL/accuracy/OOV/coverage/vocabulary/parameter/runtime/memory plots, training/validation histories, and best/last epoch annotations. For one seed report `n=1`, no variability estimate or uncertainty band; distinguish patience stop from epoch cap and subset aliases from distinct observations.
- [ ] T018 [Plan:3.2] Document keys/units/nulls/infinite statuses/denominators and metric formulas in exported `schema.json` and `DATA_DICTIONARY.md`; expose run/data/vocab/event hashes in every joinable table. Verify both notebook compatibility exports on the same tiny models/data and extend `src/tests/test_reporting.py`; reports must rebuild from saved artifacts without training.

### 4.1 Verified Colab Packaging And Execution

Requirements: REQ-008, REQ-009, REQ-011. Inputs: new protocol, complete prepared snapshot, existing Colab helper.

- [ ] T019 [Plan:4.1] Extend `src/lm/colab.py` to bundle code, pinned config, provenance, all split/budget manifests and actual sentence data, comparison helpers, and environment specification, with a member SHA-256 manifest and overall bundle digest. Check all six corpora are eligible and prepared before accepting a full-sweep bundle.
- [ ] T020 [Plan:4.1] Generate `colab/data_efficiency_comparison.ipynb` as valid notebook JSON, with `metadata.language` on cells and retained `metadata.id` for any existing edited cells. Include digest verification, Drive mounting, package/environment capture, preflight, train, seal, evaluate, report, verification, and export steps. Transfer the verified bundle into private Drive; browser authorization may require the user.
- [ ] T021 [Plan:4.1] Preflight sentence/logit limits, counts memory, GPU suitability, disk/Drive space, and projected artifact sizes. Serialize neural jobs on one GPU, synchronize timing, checkpoint persistently, verify completed jobs on resume, and record failure status. Never silently cap vocabulary, truncate sentences, downsample a corpus, alter a model, or skip a required condition to fit resources.

### 4.2 Pilot, Full Sweep, And Local Verification

Requirements: REQ-006, REQ-008, REQ-009, REQ-010, REQ-011, REQ-012. Inputs: verified bundle/notebook and comparison parity checks.

- [ ] T022 [Plan:4.2] Run a tiny CPU nine-model integration fixture and a separate Colab Inaugural 20% pilot with all nine variants, seed 42, plus a resource probe for each largest corpus. Check score/report parity, probabilities, hashes, event-derived totals, checkpoint reload, and resume/tamper rejection. Use validation/resource evidence only for readiness; keep pilot outputs outside the final matrix.
- [ ] T023 [Plan:4.2] After all gates pass, upload frozen final inputs and execute all six corpora/five budgets/nine models from fresh starts; seal, score fixed validation/test data, and build reports. Require 270 completed final jobs and 540 unique split/model metric keys, all nine models at each corpus/budget, no excluded corpus, and no budget/resource failure. Record single-seed and changing-vocabulary interpretation limits.
- [ ] T024 [Plan:4.2] Export checksummed split archives containing actual training/evaluation inputs, raw/source provenance, code/config/environment, weights/counts, histories/logs, aggregate tables and full events; retrieve them locally, verify member hashes/ZIP integrity, reload a neural and count checkpoint, reproduce their metrics and selected original occurrences, then update `colab/README.md` and `docs/research_directions/data_efficiency/README.md` with verified locations and remaining limitations.

## Output Contract

Proposed shared layout, with actual bytes packaged rather than absolute-path-only references:

```text
results/comparison_sweep_v1/
  protocol.json, matrix.json, environment/, seal.json, verification.json
  source_manifest.json, input_bundle.zip, input_bundle.zip.sha256
  prepared/<corpus>/
    sentences.sqlite, manifest.json, source_documents.jsonl, splits.json
    train.txt, valid.txt, test.txt, leakage_audit.json
  <corpus>/budget_020/  # also 040, 060, 080, 100
    train.txt, documents.json, vocab.json, vocab_counts.csv, data_stats.json
    <corpus>_comparison_metrics.csv, <corpus>_comparison_examples.csv
    final/<model_id>/seed42/
      checkpoint, config.yaml, result.json, status.json, history.csv
      validation_metrics.json, test_metrics.json
      validation_events.jsonl.gz, test_events.jsonl.gz, attempts/
  runs.csv, metrics.csv, summary.csv, diagnostics.csv, examples.json
  comparison_metrics.csv, comparison_examples.csv, split_audit.csv
  schema.json, DATA_DICTIONARY.md, report.html, archive_manifest.json
```

Sharing an immutable split or training file across nine runs is acceptable if every run references its content hash and the same export contains those bytes. Raw tokens and stable occurrence maps must survive normalization and unknown-word mapping. Keep upstream archives/provenance in the private research archive, subject to verified source terms; do not commit corpora, weights, bundles, or bulky results to Git.

The compatibility batch metrics retain `split`, `model`, `family`, `accuracy`, `perplexity`, `status`, `correct`, `events`, `zero_probability_events`, `context_coverage`, `oov_rate`, `training_seconds`, and `evaluation_seconds`. Compatibility examples retain `split`, `model`, `prefix`, `expected`, `predicted`, and `log_probability`. Add corpus/budget/seed/run/hash keys without removing these columns. The incremental view also retains `correct_word_predictions`, `word_tokens`, `perplexity_events`, `sentence_end_targets`, and `train_seconds`. Explicitly distinguish its word-only coverage from legacy batch coverage.

## Validation And Execution Gates

1. Source gate: all six current source snapshots available, story/document provenance recovered, parser failures reviewed, no writes to old data/results. ROCStories remains blocked until its archive/revision is supplied or obtained legitimately.
2. Data gate: all partitions nonempty; grouped source IDs disjoint; no normalized exact sentence overlap after declared quarantine; near-duplicate findings reviewed under a declared rule; nested budgets and aliases correct; shared per-budget vocabulary is train-only; hashes cover actual prepared bytes.
3. Model gate: both backoff orders normalize on the reproduced full-support case and ordinary seen/unseen contexts; all smoothers respect target support; MLE infinite PPL remains a valid result; FFNN contexts cannot overwrite each other; every random seed is 42.
4. Comparison gate: shared input/checkpoint checks reject mismatches; identical model/data inputs give identical deterministic counts/statuses and numerically matching NLL/PPL/accuracy in notebook and automated paths; persisted event streams reconstruct metrics and OOV/frequency groups. No timing equality is required.
5. Replay/resource gate: input bundle and manifest verified; checkpoint reload/resume/tamper checks pass; sentence/logit/count-memory/disk limits pass for every corpus; hardware/determinism limitations disclosed. Bitwise cross-hardware neural identity is not promised merely because the seed matches.
6. Final gate: 270 complete final fits, 540 complete split records, all expected identities unique, no hidden exclusions, report regeneration succeeds, local archive verification and representative score replay pass. Statistical fitting/history records are not misrepresented as separately rescored training-set metrics.

## Open Dependency And Boundaries

ROCStories source archive and release identity are absent locally. The current flattened exports cannot satisfy a fresh whole-story experiment. This is an acquisition dependency, not permission to substitute another dataset, use old exports as current data, or omit the corpus.

Source access, Drive authorization, GPU quota, and corpus resource measurements are runtime gates, not verified capabilities. Runtime/storage estimates will come from fresh-data preflight and the pilot, not old three-epoch timings. This plan does not start downloads, training, or a browser session by itself.

Single-seed results support matched model comparisons on this frozen split, not uncertainty over initialization or samples. Growing per-budget vocabularies change OOV mapping, output support, and neural parameter counts; cross-budget PPL measures end-to-end data/lexicon growth, not a fixed-target intervention. Compare models within corpus/budget and retain vocabulary/OOV diagnostics beside every curve. Do not pool corpus perplexities into one leaderboard.

## Requirement Mapping

| REQ ID | Description | Plan Items | Planned Implementation Evidence |
| --- | --- | --- | --- |
| REQ-001 | Fresh six-corpus local inputs | 1.1 | Refresh driver, source manifests, pipeline logs/archives |
| REQ-002 | Fixed audited document splits | 1.2 | Preparation manifests, grouping/quarantine/overlap audits |
| REQ-003 | Five reproducible nested budgets | 1.2, 2.2 | Document lists, actual text, budget/vocabulary hashes |
| REQ-004 | Nine independent model variants | 2.1, 2.2 | Suite adapters, new config, model/context provenance |
| REQ-005 | One seed value 42 | 1.2, 2.1, 2.2 | Config validation, manifests, final paths and RNG state |
| REQ-006 | Identical validation/scoring contract | 2.1, 3.1, 4.2 | Shared evaluator/checks, normalization/parity/replay evidence |
| REQ-007 | Comparison-compatible reporting | 3.1, 3.2 | Compatibility CSVs, incremental tables, plots and HTML |
| REQ-008 | Every input and run reproducible | 1.1, 1.2, 2.2, 3.2, 4.1, 4.2 | Actual split/budget bytes, checkpoints, histories, source/config/environment archives |
| REQ-009 | Automated Colab and safe resume | 4.1, 4.2 | Verified bundle/notebook, status/attempt/checkpoint records |
| REQ-010 | Downstream reusable aggregates/events | 3.1, 3.2, 4.2 | Stable event IDs, full compressed event streams, schema/data dictionary |
| REQ-011 | Correctness/resource/pilot gates | 1.1, 2.1, 2.2, 4.1, 4.2 | Fail-closed preflight, focused checks, resource/pilot evidence |
| REQ-012 | Complete verified local results | 4.2 | Matrix audit, 270 fits/540 records, archive and replay verification |