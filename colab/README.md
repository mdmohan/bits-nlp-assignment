# Colab Data-Efficiency Pilot

[nltk_data_efficiency.ipynb](nltk_data_efficiency.ipynb) runs the existing model
implementations against audited Gutenberg, WebText and Inaugural inputs on Colab.
No access token or password belongs in this notebook or in Git.

## Frozen Comparison Sweep (2026-11-02)

[comparison_sweep_frozen.ipynb](comparison_sweep_frozen.ipynb) uses the teammate's
full-run controller with newly downloaded, locally prepared inputs. All model
fitting, including statistical baselines and fitting regression tests, must run
in a Colab GPU runtime. Local preparation never calls a language-model fit.

The new snapshot is under `results/comparison_refresh_20261102_v1`. Brown,
Gutenberg, Inaugural, Reuters, ROCStories and WebText are prepared. ROCStories
was downloaded from the public `hamishivi/ROCStories` Hugging Face mirror at
revision `cd2b3893a6ee30afada293d5a8f877c3023530e1`. Its dataset card declares
CC BY 4.0; this is a mirror-card declaration, not independent verification of
the upstream license. All 98,161 source stories are preserved as documents;
the mirror's partitions are recombined and repartitioned using the existing
70/20/10 ratios and seed 42. Identical complete stories stay in one partition.
Pinned source files, hashes, row identities and preprocessing are archived.
Exact normalized cross-split sentence overlap is zero after a recorded
test-first quarantine. The exhaustive sentence near-duplicate audit found two
Gutenberg pairs and six Reuters pairs at token-five-shingle Jaccard >= 0.9
with at least ten tokens and five distinct shingles. The user accepted these
flags as audit-only rather than training blockers. Their evidence is retained;
ROCStories was not near-scanned under this policy. Shorter and lower-similarity
pairs are outside the audit's claim.

The existing seeds, split seeds/ratios, vocabulary thresholds, architectures,
20-epoch ceiling and patience three are preserved. Budgets are 20/40/60/80/100%
prefixes of sentence lines, not document percentages. Four corpora retain
seeds 13/42/2026; ROCStories and WebText retain seed 13. The complete matrix
would contain 210 neural fits and 180 statistical conditions. One trigram
count table supports the six statistical views per corpus/budget. Vocabulary
and prediction support change across budgets, so cross-budget perplexity is
not a fixed-support comparison.

The private bundle contains the actual working-tree code, source archives,
cleaned/parsed documents, split membership and sentence occurrence provenance,
audits, rare-sequence inventories, configs and member hashes. Frozen-input
configs prevent Colab from regenerating splits. Setup verifies every bundle
member and split manifest. Keep the notebook's bundle SHA-256 pin synchronized
whenever rebuilding the bundle. V3 contains all six corpora and 8,492 verified
members; its SHA-256 is
`2ae43095c6e7f0fc95468b13a40c9d49b9004c7579bb5b961b2f7f6e60bbfff8`.
V1, V2 and their logs remain preserved separately. The five earlier prepared
corpus manifests are unchanged from V2.

```sh
PYTHONPATH=src ~/.venvs/nlp-lm/bin/python -u src/prepare_mega_run.py --output results/comparison_refresh_20261102_v1 --corpora roc-stories --audit --accept-near-duplicates --bundle colab/comparison_sweep_frozen_v3.zip
```

Notebook outputs persist in `MyDrive/NLP/comparison_sweep_20261102_v3`, with
separate `smoke/runs` and `full/runs` directories. Inaugural smoke uses 20/40%
and all existing seeds: 18 neural fits and 12 statistical conditions. Full
execution requires passing regressions, Colab resource probes and verified
smoke artifacts. Resume does not silently change settings or reuse unsigned
final weights. Preserve the bundle, configs, environment log and all output
files. Detailed full-run output is flushed to Drive while notebook output
shows pipeline milestones.

Reports retain the legacy CSVs/plots and add serialized count tables, complete
compressed base-split scoring events (including EOS, raw/mapped OOV targets,
occurrence positions and log probabilities), event-to-metric reconciliation
and an artifact hash inventory. Word accuracy excludes EOS; perplexity includes
one EOS per sentence. Zero-probability MLE events remain explicit rather than
being clamped. Fitting regressions and real smoke results are not locally
certified by the preparation checks.

Verified on a Colab T4 on 2026-11-02: all 48 regression tests passed, followed
by the complete Inaugural smoke run. Post-run verification rehashed 96 artifact
files, confirmed 18 final neural checkpoints and 18 nonempty training-history
CSVs, and reconstructed 2,559,525 scoring events across 90 base conditions.
The 12 statistical conditions and both serialized count tables are present.
The verification cell saves `smoke_verification.json` beside the durable logs
in the V2 Drive directory. V3 subsequently passed all 55 regression tests in
17.106 seconds and all 18 disposable Colab-only resource probes: one training
batch and the longest-sequence evaluation shape for each architecture/corpus
at full-budget vocabulary. Peak reserved GPU memory was below 2.0 GiB. These
probes are memory checks, not convergence or complete runtime estimates.
V3 smoke verification passed: 96 artifact hashes, 18 neural checkpoints,
18 nonempty histories and 2,559,525 scoring events across 90 base conditions.
The full six-corpus sweep has started on Brown at the 20% training budget,
with one worker and resumable Drive outputs. This confirms launch, not full
completion; Colab runtime and quota limits still apply.

### Historical CPU Comparison Attempt (Stopped)

This stopped attempt is superseded by the active Kubeflow CUDA recovery below.
Do not run Cell 8 for the current recovery workflow.

The artifact audit below confirms that neither corpus has a complete five-budget
neural archive. The earlier attempt used only
Cell 8, the comparison code cell, with hardware accelerator **CPU**,
without Run all or the original setup/full-training cells. It scores the saved
Gutenberg and Inaugural models
from `full/runs` at 20%/40%/60%/80%/100%; it never reads `smoke/runs`.
All neural scoring is forced to CPU with `--device cpu` and GPU visibility
disabled. No neural training, regression suite, smoke run or
resource probe is invoked. Brown, Reuters, ROCStories and WebText are untouched.

The cell restores the original V3 bundle into a fresh Colab runtime if needed,
installs required dependencies and reuses the original Drive run location.
It checks that all expected checkpoints and training snapshots exist; missing
artifacts stop the comparison rather than triggering retraining. The original
run must be accessible and the same public PyTorch release available. CPU/CUDA
wheel suffixes may differ (for example, `2.11.0+cpu` versus `2.11.0+cu130`);
training and comparison versions are both recorded. Different patch, minor or
prerelease versions are rejected.

The frozen comparison script has a CUDA availability guard. The cell creates
a runtime-only CPU-compatible copy using Python's AST parser, changing only
that guard and its error message while retaining the Colab-only check. The
original source bundle and training configurations are unchanged. The copy,
both source hashes and CPU execution settings are archived with each attempt;
comparison configurations use a separate `runtime_configs/comparison_cpu`
directory. Statistical fitting remains Colab-only.

Statistical baselines are fitted and scored on each budget's exact saved
training snapshot and vocabulary. Existing neural checkpoints are loaded for
inference only. Outputs include train/validation/test metrics, rare-sequence
scores, predictions, charts, scoring events, event verification and artifact
manifests under each corpus's `full/runs/<experiment>/metrics/` directory.

Logs, environment and plan are saved in a new
`comparison_attempts/<UTC-attempt>/` directory. Existing partial metrics are
backed up there before replacement; prior logs and neural checkpoints remain
intact. This supersedes the earlier Gutenberg-60% training recovery cell.
Execution requires a Colab CPU runtime and Drive authorization, not a GPU.

The 2026-11-03 CPU launch passed the runtime and PyTorch release checks but
stopped before scoring because the configured Gutenberg 100% full-run directory
was missing five checkpoint files: FFNN context 2 seeds 42/2026 and LSTM seeds
13/42/2026. No neural fitting, smoke runs or metrics replacement occurred.
Inaugural scoring was not reached. The complete audit below found additional
Inaugural gaps; the active GPU recovery restores all required artifacts before comparison.
The launcher does not silently omit incomplete models.

### Full-Run Artifact Audit And A100 Handoff

The read-only 2026-11-03 audit checked 548 expected artifacts and 90 neural
conditions across Gutenberg and Inaugural at 20%/40%/60%/80%/100%. It validated
frozen input manifests and hashes, exact training prefixes, vocabulary order,
configuration, result signatures, history fields, and CPU checkpoint loading,
finite tensors and strict model keys/shapes. No fitting or scoring was run.

The verified result is 45 usable checkpoints, 45 missing final checkpoints,
153 missing required training files, and one zero-byte resume checkpoint.
Gutenberg 20%/40%/60%/80% are complete. Gutenberg 100% and every Inaugural
budget have gaps. All 28 expected comparison output files are absent.

The complete file-by-file notes and A100 recovery scope are in
[the audit report](../results/artifact_audits/20261103T073151.483468Z/MISSING_ARTIFACTS.md).
[The JSON evidence](../results/artifact_audits/20261103T073151.483468Z/audit.json)
and [artifact CSV](../results/artifact_audits/20261103T073151.483468Z/artifact_status.csv)
include every expected path, status, missing component, size and available hash.
The reusable read-only auditor is [src/audit_neural_artifacts.py](../src/audit_neural_artifacts.py).

Drive copies are under
`NLP/comparison_sweep_20261102_v3/artifact_audits/20261103T073151.483468Z/`.
The earlier `20261103T072326.842669Z` audit is retained but superseded: it used
incorrect special-token/default-metadata assumptions that were corrected and
revalidated before this final report.

Use a new output root on the A100 machine. Keep the frozen V3 inputs,
budgets, seeds and scientific settings unchanged; do not use signed result JSON
alone to skip conditions with missing weights. At audit completion, no A100
training or comparison had been launched. A VM outside Colab needs a separate launcher that explicitly
adapts the Colab-only execution guard without altering the scientific protocol.

### Active Kubeflow CUDA Recovery

The user authorized Gutenberg/Inaugural recovery and evaluation on the existing
Kubeflow notebook server. The task notebook is
`/home/jovyan/cisco-cpt-1a/nlp_recovery_20261103/recovery.ipynb`.
The launcher is [src/run_kubeflow_recovery.py](../src/run_kubeflow_recovery.py).
It uses a separate output root, isolated per-condition jobs, and runtime-only
adaptation of the Colab execution restriction; frozen sources remain unchanged.

Training, budget-controller configuration, and neural comparison explicitly use
CUDA. Missing CUDA stops recovery rather than falling back to CPU. The pinned
runtime is PyTorch `2.11.0+cu126`, CUDA 12.6, and an A100 80 GB GPU; real device
allocation passed. Clean NumPy 1.26.4, SciPy 1.15.1 and scikit-learn 1.6.1 copies
were installed only in the isolated NLP environment after an inherited SciPy
import failure. The original notebook environment remains untouched.

On 2026-11-03, the first required fit, Gutenberg 100% FFNN context 2 seed 42,
reported `device cuda` and reached epoch 3. GPU observation showed 56%
utilization and approximately 11.6 GB of device memory. This confirms active
training, not completed recovery or evaluation.

The 45 missing fits subsequently completed successfully at 09:36:39 UTC on
2026-11-03. Every new signed result records CUDA, and all 45 sets of final
weights, histories and signed results are present. No complete fit was repeated.
Full recovery and evaluation remain pending verified retained-artifact assembly.

The 45 previously complete fits are retained.
Their 135 weight/history/result files passed source checksums and are being
transported read-only from Colab. Both corpora's five-budget comparison is gated
on assembly and validation of all 90 neural conditions. `COMPLETED.json` is
written only after comparison and the final artifact audit pass.

### Remaining V3 Corpora On Prayogashala

The user authorized neural training and full evaluation for Reuters, ROCStories,
and WebText on the existing Kubeflow GPU server. The isolated attempt directory
is `/home/jovyan/cisco-cpt-1a/nlp_v3_remaining_20261105`; the earlier recovery and
its completed outputs are not reused as training outputs or overwritten.

[src/run_kubeflow_sweep.py](../src/run_kubeflow_sweep.py) verifies frozen source,
config, and selected corpus hashes before running the original full controller.
It requires Kubeflow, CUDA, and PyTorch public release `2.11.0`. Only runtime
paths and the Colab execution restriction are adapted. The original V3 ZIP
and all scientific settings remain unchanged.

The matrix contains 75 neural fits and 90 statistical conditions: five budgets
(20/40/60/80/100%), FFNN contexts 1 and 2 plus LSTM, Reuters seeds 13/42/2026,
and ROCStories/WebText seed 13. Each corpus's neural training is followed by
the original evaluation, including six statistical variants, base-split and
rare metrics, plots, predictions, scoring events, and artifact manifests.

```sh
/home/jovyan/cisco-cpt-1a/nlp_recovery_20261103/nlp_venv/bin/python -u \
   /home/jovyan/cisco-cpt-1a/nlp_v3_remaining_20261105/run_kubeflow_sweep.py \
   --workspace /home/jovyan/cisco-cpt-1a/nlp_v3_remaining_20261105/frozen_workspace \
   --output /home/jovyan/cisco-cpt-1a/nlp_v3_remaining_20261105/run
```

Use `--plan-only` for checksum and routing validation without fitting. Execution
requires a fresh output directory by default. To continue an interrupted run,
pass `--resume-audit /path/to/reuters_audit/audit.json` with a completed read-only
neural artifact audit. The saved plan must match exactly, and all audited neural
files are rehashed before reuse. Audited corpora take the evaluation-only path;
untouched corpora run training and evaluation. Partial neural corpora require
separate recovery and are not silently retrained. Prior metrics and serialized
count tables are retained under `run/resume_attempts/<UTC>/previous_evaluation/`.
Do not use a new output directory to blindly repeat an interrupted run.
The launcher writes `run_plan.json`, environment and
per-corpus progress records; `COMPLETED.json` is written only after all three
full pipelines return successfully. This section documents the launch procedure,
not evidence that the experiment has completed.

Run the continuation synchronously in a Jupyter kernel, with logs on the persistent
volume. A browser disconnect is not proof of failure; check the process and output
markers. A notebook/container restart still interrupts the work and requires a
fresh artifact check before any further continuation.

## Prerequisites

- Install the local environment using the [root setup](../README.md#start-here).
- Restore the imported corpus files described in the [prepared-data guide](../data/processed/readme.txt),
  including the matching import manifest, document stores and checksum sidecars.
- Keep the prepared input bundle private. It contains corpus text as SQLite
  sentence indexes and is intentionally ignored by Git.
- Use a GPU runtime with sufficient memory and Google Drive space for several
  gigabytes of checkpoints and result archives. Colab limits still apply.

The synthetic tests and `--dry-run` do not need private corpora. Building a real
bundle does. A Git clone alone is not the complete experiment input.

## Inaugural Validation-Stopping Follow-Up

The approved follow-up uses
[data_efficiency_inaugural_convergence.yaml](../src/configs/data_efficiency_inaugural_convergence.yaml):
Inaugural only; 10%, 50% and 100% training-document budgets; LSTM and context-2
FFNN seeds 42, 43 and 44; at most 20 epochs; patience three. Statistical
baselines are refitted on the same prepared data. Original imports, test bytes
and completed pilot artifacts remain unchanged.

Preparation applies the opt-in derived sentence quarantine with precedence
test, validation, training. Exact matches use lowercase and collapsed whitespace.
Inspect the prepared manifest's exclusions, partition counts, near-duplicate
audit and vocabulary/OOV measurements before launching. This is a repaired
follow-up, not a pure epochs-only replication of the pilot. Instructor approval
is still needed before using Inaugural as the primary assignment corpus.

The local preparation audit on 2026-10-29 excluded 26 sentence occurrences.
Training retains 43 documents / 3,879 sentences / 105,863 words; validation
retains five documents / 423 sentences / 9,253 words; test retains 12 documents /
1,067 sentences / 25,821 words, byte-identical to the import. Exact overlap
between every pair of partitions is zero. The sentence near-duplicate audit
flagged no cross-split pairs at five-token-shingle Jaccard >= 0.9, considering
sentences with at least ten tokens and five distinct shingles. Shorter or
lower-similarity near-duplicates are outside that audit's claim.

Build a new matched bundle and notebook after source changes are complete:

```sh
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency_inaugural_convergence.yaml --stage prepare
PYTHONPATH=src ~/.venvs/nlp-lm/bin/python -u -m lm.colab --suite src/configs/data_efficiency_inaugural_convergence.yaml --output colab/inaugural_convergence_bundle.zip --notebook colab/inaugural_convergence.ipynb
```

Upload the generated notebook, select a T4 GPU, and run cells in order. Upload
the matching bundle when prompted and authorize Drive directly. The generated
notebook pins the bundle SHA-256, epochs, patience, budgets and seeds. It checks
those settings instead of silently overriding the bundled protocol. Results
persist in `MyDrive/NLP_Experiments/inaugural_convergence_v1/`.

The report includes each seed, sample standard deviation, best and last epochs,
stopping reason, measured cost and learning curves. A patience stop indicates a
validation plateau under the chosen rule, not mathematical convergence. A run
exhausting 20 epochs before satisfying patience is labelled budget-limited; do
not extend selected runs based on their test results. Validation NLL alone
selects restored checkpoints.

## Scoring-Only 100-Case Analysis

[analyze_inaugural.py](../src/analyze_inaugural.py) analyzes the frozen follow-up
without calling training or modifying its artifacts. Run it in the environment
and source tree preserved in the experiment's verified `input_bundle.zip`:

```sh
python src/analyze_inaugural.py --experiment /path/to/inaugural_convergence_v1 --output /path/to/inaugural_analysis100_v1 --device cuda
```

Upload the standalone script into the restored source tree; no new model bundle
or training is needed. The output must be a new directory outside the experiment.
It verifies the seal, input bundle, prepared text, checkpoint hashes and saved
test metrics, then scores all 27 final checkpoints at 10%, 50% and 100%.
Rescored NLL must agree with the archived evaluation within numerical tolerance.

The frozen selection is 30 known-word unseen combinations, 30 OOV targets,
20 local FFNN probability wins and 20 all-seed shared failures, in that order.
Every candidate has full-budget original normalized trigram count at most five;
EOS is excluded. Cases are selected by hashed stable source-event IDs with
no sentence reuse. Categories are deduplicated in order; an insufficient
category fails explicitly instead of silently substituting examples. These
outcome-stratified cases are illustrative, not an unbiased error-rate sample.

Every case retains original-cased text, normalized and mapped targets,
original/mapped training counts, model-visible contexts, and target probabilities
and top predictions from all three neural seeds and statistical baselines at
every budget. UNK correctness is not original-word recovery.

Loss decomposition separates known targets, OOV targets and EOS, and reconstructs
total loss by token-weighted NLL contributions. Additional cohorts track the
same test occurrences across budgets using original target membership, and
preceding-two-word-plus-target membership, in the three vocabularies. Stable
cohorts do not eliminate changing output support or changes in longer LSTM
prefixes. Infinite MLE losses and zero-event counts remain explicit.

Start with `report.html`; machine-readable artifacts include `cases.json`,
`case_scores.csv`, `case_counts.csv`, per-seed `decomposition.csv`, seed-aggregate
`decomposition_summary.csv`, `events.csv`, compressed per-event `scores.npz`,
`protocol.json`, `verification.json` and a checksummed artifact manifest.

## Exhaustive Rare-Target Evaluation

[analyze_rare_targets.py](../src/analyze_rare_targets.py) extracts **every**
matching held-out occurrence from the verified all-event scores, with no sample,
top-k or example limit:

```sh
python src/analyze_rare_targets.py --experiment /path/to/inaugural_convergence_v1 --analysis /path/to/inaugural_analysis100_v1 --output /path/to/inaugural_rare_targets_v1
```

Run alongside the existing standalone
[100-case analyzer](../src/analyze_inaugural.py) in the verified source/runtime
environment. `--analysis` must be the full analysis directory on Drive, not the
portable local reader: its original `artifact_manifest.json`, `scores.npz`,
`events.csv` and prepared index are required. The new output must not be inside
either input directory.

The approved definitions are original word training frequency 1-5 and original
trigram training frequency 1-5, with frequency-zero events labelled unseen
separately. Counts are budget-specific, training-only and computed before UNK
mapping, using the frozen whitespace/lowercase preprocessing. EOS is excluded.
Both definitions are reported separately because they overlap.

The lexical inventory includes all rare training word types plus observed unseen
held-out types. Rare training words absent from test are explicitly labelled
`no_heldout_context`, not assigned fabricated evaluation scores. The combination
inventory includes all rare/unseen trigram types occurring in held-out word
events. Each original target occurrence keeps its full sentence context, mapped
LSTM prefix and context-2 FFNN input.

Every model/budget/seed result comes from previously executed frozen-checkpoint
inference, verified by hashes and original-event alignment. All 27 word-only
losses, MLE zero counts and correct counts must reproduce the prior analysis.
No retraining or duplicate inference is needed. The existing interactive CLI
evaluates complete text but has no exhaustive original-frequency filter; the
legacy rare-ngram utility samples candidate sentences and uses a different
tokenizer, so neither is substituted for this audited workflow.

Start with `report.html` and the searchable `rare_targets.html`. Complete lists
are in `lexical_inventory.csv` and `combination_inventory.csv`; exhaustive
per-type results are in `target_results.csv.gz` and
`combination_results.csv.gz`. Every eligible occurrence and its model/seed
predictions/probabilities are in `event_metadata.csv.gz` and
`event_scores.csv.gz`. `runs.csv`, `summary.csv`, `paired_comparisons.csv`,
`coverage.csv`, protocol, verification and checksums document coverage.

Mapped accuracy can credit predicting UNK; lexical accuracy never credits UNK
as recovery of an original OOV word. NLL/perplexity still score mapped targets.
Infinite MLE losses remain explicit. Seed standard deviations measure
initialization variation, not dataset uncertainty.

## Build A Matching Bundle

From the repository root:

```sh
~/.venvs/nlp-lm/bin/python src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage all --dry-run
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage prepare
PYTHONPATH=src ~/.venvs/nlp-lm/bin/python -u -m lm.colab --suite src/configs/data_efficiency_nltk_pilot.yaml --output colab/nltk_pilot_bundle.zip
```

Preparation preserves imported test membership and derives validation only from
existing training documents. The explicit WebText repair keeps test bytes
unchanged and excludes overlapping training source groups in a separate derived
directory. A blocked corpus is not silently replaced with a sentence-sampled run.

The packager writes `nltk_pilot_bundle.zip` and a `.zip.sha256` sidecar. It includes
source/tests, the selected corpora's prepared manifests and sentence indexes,
configuration, repair audit and `bundle_manifest.json`. Individual bundle members
have SHA-256 hashes. It does not include original corpus databases or checkpoints.

The checked-in pilot notebook's `EXPECTED_BUNDLE_SHA256` identifies the archived
pilot bundle, not every future build. **Use `--notebook` to generate a new
matching notebook after rebuilding.** ZIP metadata or source changes can change the archive hash even when
settings look similar. Do not bypass the check or assume an old bundle contains
newly edited code.

If source or settings changed since a previous preparation, select a new `name`
and `output_root` in a copied suite configuration before running it. Supply that
same configuration to both commands. Completed output identities are immutable.

## Run The Notebook

1. Upload the notebook to Google Colab and select **Runtime > Change runtime type
   > T4 GPU**.
2. Set `EXPECTED_BUNDLE_SHA256` to the bundle's sidecar hash. Keep `EPOCHS = 3`,
   `PATIENCE = 3`, ten `BUDGETS`, `SEEDS = [42]` and the three selected corpora for this pilot. Use a new
   `OUTPUT_NAME` for a new bundle, changed settings or a new experiment.
3. Run the upload cell and select exactly one prepared ZIP. The notebook verifies
   the outer hash, safe extraction paths, member hashes and corpus selection.
4. Run the Drive cell and complete Google's authorization directly in the
   browser. Keep `USE_DRIVE = True` for durable outputs. Without Drive, results
   under `/content` disappear when the runtime is discarded.
5. Run the stage cell: `train`, `seal`, `evaluate`, then `report`. It saves combined
   output to `colab_execution.log` and stops on a failed stage.
6. Run the preview and export cells. The export writes an artifact hash manifest,
   creates a ZIP outside the result folder, retains it beside the result folder
   on Drive, and requests a browser download.

The default Drive location is
`MyDrive/NLP_Experiments/<OUTPUT_NAME>`. The archived completed run is
`nltk_data_efficiency_pilot_v2`; do not reuse that directory for changed code.

## Scientific Settings

| Setting | Pilot |
| --- | --- |
| Training budgets | 10%, 20%, ..., 100% of whole training documents, nested and seeded |
| Validation/test | Fixed raw populations per corpus; test scoring follows sealing |
| Vocabulary | Training-only, fresh per budget, `min_freq=2`, shared within that budget |
| Models | `bigram_mle`, `trigram_mle`, `trigram_addk`, `lstm`, `ffnn` |
| Add-k selection | Validation grid: 0.01, 0.1, 1.0 |
| Neural training | Three epochs, final seed 42, fresh model at every budget |
| Prediction targets | Word tokens plus EOS; PAD/BOS excluded |
| Matrix | 150 final fits, including 60 neural fits, plus 90 add-k tuning fits |

The FFNN uses two preceding words; the LSTM uses the sentence prefix. A saved
best-validation `.pt` contains weights only, not optimizer state. WebText's
rounded budget aliases and retained within-source variants must remain labelled.
Cross-budget perplexity is not a fixed-vocabulary intervention.

## Resume And Failures

Rerun the same notebook cells with the same bundle, settings and `OUTPUT_NAME`
after a recoverable interruption. A replacement runtime needs the upload and
Drive cells again. Complete fits are skipped only after their fingerprints and
artifact hashes pass; incomplete fits restart in new preserved attempt folders.
This is not exact mid-epoch optimizer resume. Environment changes can invalidate
a fit identity; use a new output directory rather than forcing reuse.

Check `colab_execution.log`, per-run `status.json` and `attempts/*/run.log` for the
failing stage. Do not change seeds, resource limits or model sizes inside an
existing run to hide an error. For quota/authentication failures, restore access
or start a separate eligible runtime; repeated automatic retries cannot remove
Colab limits. Increasing cached GPU usage alone does not establish a memory leak.

If Colab's output JavaScript fails, inspect the durable log through Drive. A blank
preview does not prove training failed. The completed pilot used a direct Drive
folder download because preview/export controls were unreliable.

## Retrieve And Verify All Results

Download the exported ZIP, or download the entire result folder through Drive.
Drive can split a folder into `-001.zip`, `-002.zip`, `-003.zip`, etc. Keep every
part, wait for the downloads to finish, and inspect each outer ZIP index before
extracting. File existence or `zipfile.is_zipfile()` alone is insufficient during
an active transfer: nested PyTorch checkpoint ZIPs can cause false positives.

Extract under a new `results/colab_imports/<timestamp>/` directory, never over a
previous run. Reject unsafe paths, symlinks and duplicate file names. Check ZIP
CRCs, the artifact manifest if present, sealed artifact hashes from `seal.json`
and each run's `status.json`, plus test hashes in `evaluation_status.json`. Preserve
the original ZIPs; verification does not require unpickling model/count files.

Expect 150 final `result.json`, validation metrics and test metrics, 90 tuning
results, 60 neural histories/checkpoints and all budget vocabularies. Keep the
input bundle: its prepared sentence indexes and manifests preserve the exact
train/validation/test data. Histories and fitting statistics are included;
separate final training-set rescoring was not performed.

Open the extracted `report.html` with its relative assets intact. Table and CSV
model order is `bigram_mle`, `trigram_mle`, `trigram_addk`, `lstm`, `ffnn` within
each budget. See the [source guide](../src/README.md#saved-artifacts) for artifacts
and the [batch report](../docs/research_directions/data_efficiency/batch-report.yaml)
for the verified pilot. Generated results and private bundles are not pushed.