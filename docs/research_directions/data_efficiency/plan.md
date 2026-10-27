# Implementation Plan: Automated Data-Efficiency Experiments

Date: 2026-10-26

Status: The three-corpus Colab pilot completed training, sealing, held-out evaluation and reporting. Its local import passed ZIP CRC checks, 2,434 sealed/evaluation artifact hash checks and all 32 preserved input-bundle member hashes: 150 final runs, 90 add-k tuning fits and 60 three-epoch neural histories. The approved pilot uses Gutenberg, WebText and Inaugural, with neural seed 42 at each 10%-100% document budget. Original imported files are unchanged. This does not complete the broader research rollout.

Verified local results: [offline report](../../../results/colab_imports/20261026T175208Z/nltk_data_efficiency_pilot_v2/report.html) and [import verification record](../../../results/colab_imports/20261026T175208Z/verification.json). Training histories and fitting statistics are included; separate final training-set rescoring was not performed.

Approved WebText amendment (2026-10-26): keep test bytes unchanged, exclude every overlapping training source group, and retain raw/cleaned variants within source groups. Derived files and the audit are separate. Only two training groups remain after validation, so rounded aliases are explicit. The seven-corpus research rollout below is a broader, partially implemented target, not a claim of completed experiments.

Specification: [Direction 1](../01_data_efficiency.md), extended by the user's request to evaluate the seven existing corpora automatically.

## Summary

Build one configuration-driven experiment suite around the existing neural trainer and statistical n-gram implementation. For each eligible corpus, freeze document partitions, construct nested training budgets, build a vocabulary from each budget's training subset, train each model independently, and generate metrics and plots that account for vocabulary growth.

The training population and its vocabulary can grow. Raw validation/test populations, preprocessing, vocabulary-construction policy, and scoring rules remain fixed within a corpus; mapped targets and the prediction alphabet can change between budgets. All models at a given budget share its vocabulary. A larger budget is a fresh fit, not continuation of the smaller model.

Run all seven registered corpora through preflight, but do not silently turn an ineligible corpus into a sentence-sampling experiment. Preserve its blocked status and reason in the suite report. Different corpora are separate replications, not one pooled perplexity leaderboard.

## Planning Inputs And Constraints

- Direction 1 is the scientific specification. Its suggested Brown/WikiText dataset is adapted to the user's existing seven-corpus collection. The user's selected `per_budget` vocabulary policy replaces the earlier fixed-smallest proposal: this is an end-to-end data-growth study, not a fixed-lexicon intervention.
- The four read-only audits covered neural implementations, data provenance, scoring/reporting, and orchestration. Decisions below use source inspection and the small runtime checks recorded here, not every unverified recommendation in those audits.
- No migration constitution, formal REQ identifiers, architecture artifact bundle, or project topology was supplied. This is an experiment-suite extension, not a migration. The REQ identifiers below are local planning labels extracted from Direction 1 and the user request, not additional upstream requirements.
- Preserve existing source data, historical results, commands, and public APIs where possible. New suite behavior is explicitly versioned and opt-in.
- The user separately authorized moving the supplied corpus artifacts into the pipeline's actual output layout. This import is complete and recorded in the checksummed import manifest; it does not authorize regenerating splits or rebuilding document stores.
- No dataset downloads, corpus repairs, duplicate removals, or production training are authorized by this planning document alone.

## Verified Starting Point

| Evidence | Consequence |
|---|---|
| [The current controller](../../../src/lm/controller.py#L29) builds a new vocabulary from each supplied training set. | Retain training-subset-only construction, but persist one shared vocabulary per corpus/budget; the percentage flag alone lacks the required document controls and provenance. |
| [The percentage loader](../../../src/lm/data.py#L38) selects the first fraction of sentence lines. | Replace it for this suite with persisted, seeded whole-document budgets. Keep the legacy flag for exploratory runs. |
| [The controller](../../../src/lm/controller.py#L165) always invokes data preparation. | Add a prepared-data path; the suite must not regenerate the supplied exports implicitly. |
| [The pipeline](../../../src/data-pipeline/data_pipeline.py#L243) defaults to `data/processed/<corpus>/`, with `<corpus>.db`, `train.txt`, optional `valid.txt`, `test.txt`, and `rare-ngrams.csv`. | Use the imported canonical paths below, not the former import directory or raw downloads. |
| Six supplied stores have a readable SQLite `Dict(key, value)` schema, but Python 3.12 `dbm.whichdb` cannot identify them. | Implement a schema-checked read-only SQLite-shelve reader or verify a compatible interpreter before using the existing shelve reader; path/checksum success alone is not runtime compatibility. |
| [The modular trainer](../../../src/lm/neural/trainer.py#L97) sums token NLL and excludes padding. | Keep this training foundation and expose aligned token scoring. |
| [The standalone implementation](../../../src/lm/statistical/neural_lm.py#L178) already gathers per-target log probabilities. | Adapt that technique into a shared scorer, without copying its entire training loop. |
| The training reader counted 1,067 Inaugural test sentences; the CLI counted 1,072 by re-segmenting lines. | Unify sentence reading before comparing saved metrics with CLI output. |
| Rare analysis uses Treebank tokenization while model loading uses whitespace. | Existing novelty labels must not be reused as if they described identical model events. |
| Each rare CSV has 100 bigram and 100 trigram examples, all with train count zero. | Retain them as historical diagnostic reports, not exhaustive frequency groups or a third split. |
| Checkpoints are under `source_code/results/`; historical metadata is under `src/results/`. | New run bundles must keep weights, vocabulary, config, and metrics together. Do not silently rewrite old bundles. |

Runtime evidence for the neural choice: after copying identical one-layer LSTM weights and disabling dropout, both implementations scored the same six targets. Their total log probabilities differed by approximately 1.2e-7. This checks the small constructed case, not equivalence of their training defaults or production performance.

### CLI Capability Audit And Reuse Boundary

The existing [CLI](../../../src/lm/cli.py) and [README](../../../src/README.md) already cover most model-level operations. This plan is an integration and protocol extension, not a new language-model implementation. Existing capability is not the same as a completed automated experiment, and unchecked tasks below describe the remaining integration work rather than absent models.

| Capability already implemented | Existing owner | Remaining suite work |
|---|---|---|
| Configurable bigram/trigram counting, MLE, add-k and backoff | `CountNGram`, `LanguageModelCLI.load_ngram`, shared `NgramLM` | Reuse counting and encoding; dispatch deterministic fits beside neural runs and supply the explicit budget vocabulary instead of relying on active-checkpoint path matching. |
| Training-only vocabulary and shared neural/count vocabulary | `Vocab`, controller initialization, CLI checkpoint-vocabulary reuse | Persist one vocabulary per document budget, with source/hash checks and explicit injection across every model/seed. Do not invent a second vocabulary builder. |
| FFNN context control and sentence-prefix LSTM | Existing registered neural models and `run.py` overrides | Reuse their model constructors unchanged except where the shared target-support contract requires an explicit interface extension. |
| Neural seeds, Adam, early stopping and best-validation restoration | `LanguageModelController.run`, `Trainer.fit` | Reuse the training loop; permit fit/validation without automatic final-test evaluation until the suite is sealed. |
| Token probabilities, full-file PPL, OOV, top-1 accuracy, n-gram coverage and zero events | `LanguageModelCLI.evaluate`, `_sentence_events`, `_neural_scores` | Extract numeric result/event accumulation from terminal rendering, preserve the existing counters, align readers/support and batch full sentences efficiently. `evaluate` currently prints rather than returning a structured result. |
| Neural checkpoint loading, predictions and side-by-side comparison | `load_neural`, `predict`, `resolve_neural`, `compare` | Reuse loading/rendering contracts and keep checkpoint/config/vocabulary together. `compare` currently selects one n-gram and one neural model; the suite aggregates the complete matrix. |
| Training histories, timing, checkpoints, vocabulary JSON, run CSV and seed summaries | Controller output and pandas aggregation | Extend existing artifacts with corpus/budget/provenance IDs, statistical rows, consistent timing, collision-safe paths and verified resume; do not replace them with an unrelated result framework. |
| Document-level splitting, database checksums and rare-case discovery | Existing data-pipeline helpers | Reuse safe readers/contracts/audit helpers while preserving imported test membership; add backend compatibility and nested budget manifests, not a new download/cleaning pipeline. |

The remaining work is to connect these pieces through a callable batch path, not script the interactive prompt or scrape its printed tables. Important boundaries remain: `--train-percent` selects sentence-line prefixes; the CLI evaluator re-segments lines; neural `predict` masks PAD/BOS but `_neural_scores` and the trainer do not; `CountNGram.valid_target_ids` restricts prediction candidates but does not pass an allowed-target set to statistical smoothing. Equal vocabulary lists alone do not resolve these event/support differences. The CLI also batches repeated full prefixes by length, which is correct for context but needlessly repeats neural work and can retain substantial prefix data on large inputs.

The [README](../../../src/README.md) describes CLI baselines and event scoring but its former final checklist incorrectly called them absent. Interpret the missing feature as automated batch integration and persisted analysis. Existing `runs.csv`/`summary.csv` generation is also implemented; new graphs, combined statistical/neural tables, frequency-group analysis and verified resume remain extensions.

## Corpus Intake And Eligibility

Paths below are relative to the repository root. The authoritative imported data root is `data/processed`, matching the existing pipeline default. All seven corpus folders now contain `train.txt`, `test.txt`, and `rare-ngrams.csv`; filename normalization did not change their bytes.

| Corpus | Document database inside `data/processed/<corpus>/` | Observed grouping evidence | Initial action |
|---|---|---|---|
| `blog-posts` | `blog-posts.db` | SQLite store has 19,320 serialized records. Source XML processing preserves one output file per source XML; author grouping still needs verification. | Recover source/author IDs and reconcile records with exports using bounded reads. |
| `gutenberg` | `gutenberg.db` | SQLite-backed shelf has 18 source-file keys. | Recover book IDs; verify membership in existing exports; flag coarse document budgets. |
| `inaugural` | `inaugural.db` | Shelf has 60 source-file keys. | Recover speech IDs and verify alignment; use for an initial validation-only pilot. |
| `movie-reviews` | `movie-reviews.db` | Shelf has 2,000 source-file keys with positive/negative source directories. | Recover review IDs. Candidate primary corpus after provenance checks and assignment distinctness approval. |
| `reuters` | `reuters.db` | SQLite store has 10,788 serialized records, verified after decompression. | Inspect keys/record structure and reconcile exports; verify article rather than multi-article container granularity. |
| `rocstory` | Missing; configure `database: null` | Sentence exports alone do not establish story membership. | Recover original story IDs and export mapping. Otherwise block the primary document-budget study. |
| `webtext` | `webtext.db` | Shelf has 12 keys: raw and cleaned versions of the same six source filenames. | Block until variant grouping, duplication, and split contamination are resolved explicitly. |

All seven lack a supplied validation file. Counts above were verified by read-only SQLite queries; they do not prove that current train/test membership is document-disjoint. No full-corpus duplicate or membership audit was performed.

### Completed Import And Document-Store Contract

- The pipeline writes document-preserving records through `data_process.py` as `db[absolute_source_path] = parsed_paragraphs`, including sentence lists. The `.db` store, not a flattened train/test file or a rare-sequence CSV, is the available source of document grouping. Embedded absolute keys were preserved; preparation must derive stable relative IDs rather than rewriting original stores.
- Seven folders and 41 existing files were moved from `nlp_corpus` into `data/processed`. Their original/destination paths, sizes, and SHA-256 values are recorded in [the import manifest](../../../data/processed/corpus_import_manifest.json). All moved files matched the baseline hashes, including existing SQLite sidecars, cleaned content, and audit files. The old directory is no longer a corpus source.
- Legacy `train_sentences.txt` / `test_sentences.txt` names became `train.txt` / `test.txt`; `rare.csv` / `rare_ngrams.csv` became `rare-ngrams.csv`. Existing standardized names were retained. No sentence text or split membership was regenerated.
- Original compressed stores remain at `data/processed/blog-posts/archives/blog-post.db.gz` and `data/processed/reuters/archives/reuters.db.gz`. Verified streaming decompression produced `blog-posts.db` (1,516,687,360 bytes) and `reuters.db` (24,059,904 bytes). Archives are outside the database filename prefix scan, so they cannot masquerade as dbm backing files.
- All six available document stores have verified `<corpus>.db.sha` sidecars generated with `database_integrity.write_database_sha`. These use the pipeline's backing-file checksum convention; they are distinct from the plain file hashes in the import manifest. Future preparation must verify both provenance and database integrity before reading.
- Read-only schema/count inspection succeeds for all six stores using SQLite. The experiment environment is Python 3.12.14, whose `dbm` cannot recognize these SQLite-backed shelves. Implement and test a read-only, schema-checked adapter for trusted local serialized records, preserving the existing document/sentence contract, or explicitly verify a compatible interpreter. Do not convert or overwrite the original databases as an implicit workaround.
- The current orchestrator checks its download registry before cache reuse, and the registry does not include all seven imported corpora. Its cached path can re-split exports or fall back to rebuilding/downloading on an incompatible reader. The suite's prepared-data path must bypass this orchestration, not simply point its output at the import root. Imported availability is not eligibility or training readiness.

Eligibility states are `eligible`, `needs_provenance`, `needs_repair`, and `resource_blocked`. Register all seven regardless of state. Any relaxed sentence-only study must have a separate protocol ID and must not be presented as satisfying the primary whole-document design.

## Research Protocol

### Partitions, Budgets, And Vocabulary

1. Hash and inventory the existing sources and exports before preparation. Recover stable corpus-relative document IDs; keep original paths as provenance rather than using machine-specific absolute paths as identities.
2. Reconcile document sentences with exported lines using duplicate-aware membership and order checks. Text equality alone cannot assign an ambiguous repeated sentence to a document. Block unresolved mappings.
3. Preserve the existing test population. Derive validation only from the existing training population, using whole documents. Start with 10% of training documents for validation; this is not a claim that the resulting overall allocation is exactly 80/10/10. Record actual allocations.
4. Audit normalized exact and near duplicates across candidate partitions. Use the existing MinHash/LSH machinery where suitable, with a declared candidate rule and exact similarity verification. A proposed rule is token 5-shingles, Jaccard >= 0.9 for substantial documents, with fixed short-document handling. Common short phrases are not automatically document leakage.
5. Do not edit test content to repair contamination. Remove or quarantine overlapping training-pool groups under a predeclared policy, or block the corpus for an explicitly approved new split version. Record every decision; never redraw splits to improve model results.
6. Within the remaining training pool, persist a seeded document order. Use genre/collection strata when valid metadata exists. Keep this data seed separate from neural seeds.
7. Budgets are prefixes of that document order at 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, and 100%, using documented rounding. Each run adds the next 10 percentage points of the training pool to the preceding subset, not 10% of the remaining data. Persist exact document IDs and actual document, sentence, and token counts. Every smaller budget must be a subset of every larger one.
8. Small corpora can produce identical rounded subsets at different percentages. Report these aliases explicitly and block a ten-distinct-budget claim; never pretend they are separate data points or invent smaller document units.
9. Build one vocabulary from each resolved budget's own training subset, with the same `min_freq=2` policy at every budget. Map that subset's singleton words to `<unk>`. Never use later-budget, validation, or test text to select its words or IDs. Persist deterministic token IDs, the training-subset source hash, vocabulary hash, and target-support hash; freeze them across all model families, trials, and seeds at that corpus/budget. Verify that training includes unknown-token examples; resolve a degenerate vocabulary before test evaluation.
10. Retain original tokens and budget-specific OOV flags for analysis. Escape or reject literal reserved-token collisions under a fixed rule. With nested training subsets, a fixed minimum count, and no vocabulary cap, lexical vocabulary sets must be nested and held-out OOV rates must not increase; they need not change at every step. Token IDs may differ between budgets, so never reuse encoded arrays, counts, or weights under a different vocabulary hash.

Vocabulary, frequency/count estimates, and training examples come from training text only. Validation selects smoothing/training settings and stopping checkpoints; final test text never selects them. Held-out tokens can be categorized by predeclared diagnostic rules without using model outcomes to select favorable cases.

### Selected Vocabulary Policy

Use `per_budget` as the sole vocabulary policy in the planned sweep, following the user's decision. The 10% condition uses only its 10% training subset to build vocabulary; the 20% condition rebuilds from its cumulative 20%, and so on through 100%. Build or load that budget's vocabulary once and share it across all compared models and seeds, rather than constructing model-specific dictionaries. No fixed-smallest or full-training-pool vocabulary arm is included in this matrix.

Keep the construction rule fixed, not the resulting lexical inventory. A word can move from `<unk>` to a lexical target when its cumulative training count reaches `min_freq=2`. Vocabulary size may therefore grow or remain unchanged at a step; do not force additions or admit held-out words to make growth visible. Preserve each immutable budget-specific vocabulary in its run provenance.

This measures practical data growth plus vocabulary expansion. Changing vocabulary changes `<unk>` mapping, OOV rates, prediction support, and neural parameter counts even when embedding/hidden dimensions stay fixed. A fixed raw test file therefore does not make cross-budget perplexities measurements of an identical prediction task. Report vocabulary size and OOV beside each PPL point; compare model families directly within each budget and interpret cross-budget curves as end-to-end outcomes, not isolated gains from additional observations.

Vocabulary policy, cutoff, source subset, and vocabulary/target-support hashes belong in preparation manifests, run fingerprints, metrics, and report grouping. A checkpoint or mapped-count cache must never cross vocabulary hashes. Recount the entire current subset under its current mapping: adding counts for only the new documents is incorrect when earlier `<unk>` occurrences become lexical words. Fixed-vocabulary ablations would require a separately approved protocol, not an implicit extra arm.

### Required Model Matrix

| Model ID | Configuration | Repetitions per corpus/budget |
|---|---|---|
| `bigram_mle` | Order 2, no smoothing | One deterministic final fit |
| `trigram_mle` | Order 3, no smoothing | One deterministic final fit |
| `trigram_addk` | Order 3, validation-selected add-k | One deterministic final fit |
| `lstm` | One layer, embedding 128, hidden 256; sentence-prefix context | Seeds 13, 42, 2026 |
| Optional `ffnn` | Same embedding/hidden dimensions across budgets, budget-specific vocabulary layers; explicit context, preferably 2 for trigram-style history | Seeds 13, 42, 2026 |

FFNN is useful supplementary evidence but is not required by Direction 1. A bounded-history LSTM/context sweep belongs to a separate ablation, not the minimum data-efficiency intervention. There is no requirement to force LSTM history to match a trigram for this direction.

Assuming all seven corpora pass preflight: 7 x 10 x (3 statistical + 3 LSTM seed runs) = 420 final model-result bundles under `per_budget`. Optional FFNN adds 210. Hyperparameter trials, audits, and scoring are additional work, not included in those counts. There is no second vocabulary-policy multiplier. Do not extrapolate total runtime or storage from historical Brown timings.

### Concrete Neural Implementation Decision

| Component | Decision | Reason |
|---|---|---|
| Canonical models and optimizer loop | Keep `src/lm/neural/` and its trainer. | Existing device handling, model registry, bucket batching, progress, and controller integration; both FFNN and LSTM already fit this contract. |
| Token-level scoring | Extract the CLI's existing event/metric logic and use the standalone `NeuralScorer` gather-and-unpad approach for the shared, device-aware neural scoring path. | Preserve implemented PPL/OOV/accuracy/zero-event accounting while avoiding repeated full-prefix forwards; neither a second evaluator nor a second trainer is needed. |
| Packed LSTM sequences | Optional measured optimization, not a correctness repair. | Main unidirectional padded batches are correct at non-padding targets. Benchmark packing on the selected device before changing the model contract. |
| Standalone defaults | Do not adopt automatically. | It defaults to two layers, different clipping/epoch settings, and CPU-created batches. These are different experimental settings, not evidence of superiority. |
| Standalone training loop | Keep as a reference, not a second production training path. | Duplicating production loops invites drift in tokenization, masks, device handling, and stopping criteria. |
| Architecture selection | Fix the architecture before final tests. | Choose by required semantics, measured resource use, and validation evidence, never by which implementation wins on final test PPL. |

Both implementations currently materialize encoded corpora. Neither solves the large-corpus loading problem by itself. Add disk-backed indexing and lazy batching instead of merely switching implementations or reducing batch size.

### Model Selection And Timing

- Keep embedding/hidden dimensions, layer count, optimizer family, and the search policy fixed across budgets; resize vocabulary-dependent layers for each budget and record the resulting parameter count. Proposed grids: add-k `k=[0.01,0.1,1.0]`; neural learning rate `[0.0003,0.001,0.003]`, with a dedicated tuning seed 7. Reuse the same grids and selection rule at every budget.
- Select hyperparameters on validation NLL only, then train final neural seeds independently with the selected settings. Start with Adam, 20 maximum epochs, patience 3, and explicit dropout/clipping. Record the stopping monitor and any minimum-improvement threshold rather than silently changing defaults.
- Three neural candidates at each of 70 corpus/budget conditions add 210 tuning fits. Add-k candidates can reuse immutable counts for validation selection, but record counting and selection costs separately. Final cold-fit timings must have consistent boundaries.
- Train all budgets from fresh initialization. Resume means skip verified completed runs; it does not mean warm-start the next budget. Incomplete fits restart in version 1 rather than requiring optimizer/RNG checkpoint recovery.
- Synchronize CUDA/MPS before and after timed model operations. Record preparation, tuning, fit including validation, final scoring, and report time separately. Record hardware, device, batch limits, epochs, updates, and actual tokens processed.
- Serialize neural work by default on one MPS/GPU device. Do not claim fair timing from competing concurrent jobs. Do not silently reduce precision, vocabulary, sentence lengths, or data size to recover from OOM.

### One Scoring Contract

- One prepared sentence line remains one sentence; do not re-segment it in the CLI or analyzer. Use one explicit tokenizer version, initially lowercase whitespace to match current training exports. Retokenization is a separately versioned preparation change.
- Every model at a given budget scores the same mapped target sequence: content tokens followed by one `</s>`. Raw sentence/target-position IDs and event counts stay fixed across budgets, but mapped words/IDs may change. BOS is context only; padding is never scored.
- Each budget's prediction alphabet is its lexical vocabulary plus `<unk>` and `</s>`, excluding `<pad>` and `<s>`. Apply the same budget-specific mask in neural training/scoring and the same allowed-target set in statistical smoothing. Preserve legacy defaults outside the new protocol where necessary.
- Add an explicit target-ID contract to count-model normalization; reducing an integer vocabulary size is insufficient when valid IDs are non-contiguous. Validate positive k and probability mass over exactly the allowed target IDs.
- Stream records identified by corpus, document, sentence, and target position. Neural adapters gather only true-target log probabilities and optional bounded top-k data, not the whole corpus-by-vocabulary tensor.
- Accumulate total NLL and target counts, then compute `PPL = exp(total_NLL / target_count)`. Do not average sentence or batch perplexities. Keep log-perplexity available to handle numeric overflow.
- An MLE zero probability is a valid completed result with infinite PPL, not a failed run. Report the zero-event fraction and never clamp zeros or omit difficult events. An unseen MLE context can have zero total conditional mass; do not require it to normalize like a smoother.
- Use add-k as the primary smoothed baseline. The custom `kn` recursion uses raw lower-order counts below the top order; a mass-sum test alone does not certify standard higher-order Kneser-Ney continuation semantics. KN/backoff are optional until independently validated and documented.

### Diagnostic Definitions

Use training counts of the evaluated mapped bigram/trigram: unseen is count 0, rare is count 1-2, and frequent is count >= 3. Report word-frequency bands separately rather than calling word rarity and sequence rarity the same measurement.

The main view recomputes mapped-sequence frequency groups under each budget's vocabulary and full training subset; its populations can change. A separate fixed-reference view assigns labels to raw event IDs using original-word sequence counts from the smallest budget, before unknown replacement. Keep those reference labels fixed, report group counts, and never mix original-word reference counts with current mapped-sequence counts. Fixed groups do not make prediction support identical across budgets.

At each budget, define an unseen known-word trigram only when all three original words survive that budget's vocabulary and its current training count is zero. Label OOV-containing events separately, retain original-word flags before mapping, and exclude BOS/EOS positions from this three-real-word diagnostic while retaining them as appropriate in full-test scoring. Select case IDs reproducibly using smallest-budget original-word categories and a fixed ordering/hash rule, not largest model gains; follow those same cases through changes in mapping, OOV status, and counts. Report empty categories instead of inventing examples.

## Requirements

| ID | Requirement |
|---|---|
| REQ-001 | Register all seven existing corpora with explicit files and honest eligibility states. |
| REQ-002 | Recover document groups, audit overlap/duplicates, and preserve test population with training-only validation derivation. |
| REQ-003 | Persist nested whole-document budgets with fixed evaluation populations and independent cold-start fits. |
| REQ-004 | Build vocabulary only from each budget's training subset, share it across models/seeds at that budget, and retain budget-specific hashes and original OOV information. |
| REQ-005 | Run the four required model families; keep FFNN an optional configured extension. |
| REQ-006 | Use one canonical neural implementation with justified reuse from the alternative. |
| REQ-007 | Share normalized prediction support within each budget, preprocessing, raw event alignment, and token-log-probability scoring. |
| REQ-008 | Use validation-only selection with consistent search policy and early stopping across budgets. |
| REQ-009 | Separate data/tuning/neural seeds; report neural variability and measured execution conditions. |
| REQ-010 | Record PPL/NLL, vocabulary/OOV, zero events, costs, and actual data/training counts correctly. |
| REQ-011 | Produce frequency-conditioned diagnostics, smoothing ablation, and at least ten reproducible sequence analyses. |
| REQ-012 | Generate per-corpus learning/cost/diagnostic graphs and an explorable static results report. |
| REQ-013 | Automate planning, execution, failure handling, collision-safe outputs, and verified resumption. |
| REQ-014 | Support large corpora with indexed/lazy inputs, bounded scoring, count-storage strategy, and explicit limits. |
| REQ-015 | Preserve provenance and reproducibility through data/config/code/environment hashes and immutable run bundles. |
| REQ-016 | Supply acceptance checks, a staged pilot, updated documentation, and explicit production readiness gates. |

## Technical Context

- Python/PyTorch implementation; NumPy, pandas, PyYAML, and NLTK already declared. The existing external Python 3.12 environment has PyTorch 2.14.1. The editor default Python lacks Torch/NLTK, so commands must select the experiment interpreter explicitly.
- Add Matplotlib for static charts. Reuse the existing datasketch-based duplicate machinery only after declaring its actual dependencies and recording tested exact versions. Do not install a new training framework or build a web application for this task.
- Storage: immutable imported stores/exports under `data/processed`, with original compressed stores under each affected corpus's `archives/`; versioned experiment indexes/partitions/vocabularies and run bundles under the results tree; aggregate CSV/JSON; PNG/SVG charts and offline HTML. Use SQLite/NumPy disk-backed facilities already available where suitable.
- A 563 MB training text file does not imply a 563 MB Python working set. Account for token strings, indexes, encoded arrays, vocabulary-sized logits, count tuples, continuation sets, and checkpoints.
- Long individual sentences can exceed a token-batch limit even at batch size one. Preflight must report this and block unsupported cases. Any segmentation policy must be explicit, applied consistently before freezing the protocol, and reported as a changed experiment, not an invisible truncation.

## Constitution Check

No separate constitution was supplied. The applicable constraints are the research specification, repository instructions, and user request: preserve source data; avoid test-driven selection; keep comparisons controlled; reuse existing code; distinguish proposed behavior from executed evidence. This plan follows those constraints, with unresolved corpus eligibility intentionally retained as a gate rather than marked passed.

## Applied Guidelines

The migration-guidelines lookup found no applicable source-to-target framework migration. Python fact-grounded inspection was used for code behavior, with read-only database metadata and tiny LSTM scoring checks. No migration-specific design artifacts are claimed.

## Implementation Steps And Tasks

Checked tasks have implemented paths and focused verification. Unchecked tasks may be partially implemented; the batch report records residual work. The current pilot is the three selected corpora, not the full seven-corpus research matrix.

### Phase 1: Corpus Intake And Provenance

#### 1.1 Suite Configuration And Preflight

Requirements: REQ-001, REQ-013, REQ-015. Inputs: corpus inventory, existing configuration/entry points, user automation requirement.

- [ ] T001 [Plan:1.1] Add `src/configs/data_efficiency.yaml` and schema validation in `src/lm/experiments.py`: explicit seven-corpus imported paths under `data/processed`, prepared-data mode with rebuilding disabled, import manifest, budgets, `per_budget` vocabulary source/cutoff, seeds, model grids, protocol version, primary-corpus selection, and resource limits; keep the missing ROCStories database explicit.
- [ ] T002 [Plan:1.1] Add a read-only preflight/matrix report in `src/lm/experiments.py`: imported-file and pipeline DB checksums, actual document-reader/backend compatibility, missing validation/database, provenance state, aliases, resource estimates, expected trials, and machine-readable skip/block reasons; reject unknown or ambiguous auto-discovery and never fall back to rebuilding supplied data.

#### 1.2 Recover And Freeze Document Partitions

Requirements: REQ-002, REQ-015. Inputs: `data_process.py` file-key schema, `load_processed.py` document splitting, cleaning/audit utilities.

- [ ] T003 [Plan:1.2] Add provenance intake in `src/lm/preparation.py`: implement/test a bounded read-only SQLite-shelve adapter for the imported `Dict(key, value)` schema and trusted local serialized records, reuse the document/sentence contract from `src/data-pipeline/load_processed.py`, derive stable IDs from preserved source keys, reconcile exported lines, retain import/archive metadata, and block incompatible or unresolved records without overwriting stores.
- [ ] T004 [Plan:1.2] Integrate exact/near-duplicate audits in `src/lm/preparation.py` using suitable routines in `src/data-pipeline/clean_data.py`; group WebText raw/cleaned aliases, keep audit decisions, and block unresolved cross-partition groups without mutating sources.
- [ ] T005 [Plan:1.2] Persist fixed train-pool/validation/test membership and hashes in `src/lm/preparation.py`; derive validation only from training documents and record any approved quarantine or new split version.

### Phase 2: Nested Budgets And Bounded Data Access

#### 2.1 Budget And Vocabulary Manifests

Requirements: REQ-003, REQ-004. Inputs: frozen document partitions and the selected per-budget training-only vocabulary policy.

- [ ] T006 [Plan:2.1] Build seeded nested document budgets in `src/lm/preparation.py`, with optional metadata strata, explicit rounding/alias handling, document-ID lists, and exact sentence/token counts.
- [ ] T007 [Plan:2.1] Extend `Vocab` in `src/lm/data.py` with validated save/load and prepared-vocabulary injection; build one vocabulary from each budget's training subset using the fixed cutoff, share it across models/trials/seeds at that budget, store source/vocabulary/support hashes, preserve original OOV flags/reserved-token policy, and reject later-budget or held-out sources.

#### 2.2 Indexed Inputs And Count-Storage Limits

Requirements: REQ-014. Inputs: actual file sizes and existing in-memory `SentenceDataset`/count tables.

- [ ] T008 [Plan:2.2] Add disk-backed sentence/document offsets and lengths plus lazy encoded access in `src/lm/data.py`; key encoded caches by budget/vocabulary hashes, reuse bucket batching with an explicit padded-token cap, and avoid eager whole-corpus token lists or duplicate text copies for every budget.
- [ ] T009 [Plan:2.2] Measure existing `NgramLM` counting/lookup memory and throughput first; stream each full budget under its own mapping, reject cross-vocabulary mapped-count reuse and avoid unnecessary continuation sets for MLE/add-k in `src/lm/statistical/ngram.py`. Add `src/lm/statistical/counts.py` with a disk-backed strategy only if measured resource limits require it; do not block the first small-corpus integration on an unneeded storage rewrite.

### Phase 3: Canonical Models And Shared Evaluation

#### 3.1 One Reader And One Prediction-Event Contract

Requirements: REQ-007, REQ-010. Inputs: observed CLI sentence mismatch and rare-analyzer tokenization mismatch.

- [ ] T010 [Plan:3.1] Reuse the CLI's existing whitespace `iter_sentences` semantics and the model reader when centralizing prepared-sentence reading/tokenization in `src/lm/data.py`; route CLI evaluation and `src/data-pipeline/analyze_rare_ngrams.py` through the declared suite policy without silently relabeling legacy rare CSVs.
- [ ] T011 [Plan:3.1] Extract `LanguageModelCLI.evaluate` event/counter logic and `_sentence_events` into a callable numeric-result/event core in `src/lm/evaluation.py`, leaving terminal rendering in the CLI. Preserve PPL/OOV/top-1/coverage/zero-event metrics, add stable event IDs, explicit shared target-support masks, batched true-target log probabilities, padding/EOS checks and strict zero/overflow handling; do not parse printed output or duplicate the evaluator.

#### 3.2 Neural And Statistical Adapters

Requirements: REQ-005, REQ-006, REQ-007. Inputs: neural implementation decision and existing `NgramLM` interfaces.

- [ ] T012 [Plan:3.2] Adapt the modular trainer/model path in `src/lm/neural/trainer.py` and controller injection in `src/lm/controller.py` for each budget's prepared vocabulary/lazy data and shared scoring; size vocabulary-dependent layers from its support and port the standalone gather-and-unpad scoring behavior into `src/lm/evaluation.py`, not a second trainer.
- [x] T013 [Plan:3.2] Reuse the CLI `CountNGram` encoding/fitting adapter and underlying `NgramLM` in separate fit/score dispatch beside Torch models in `src/lm/experiments.py`; accept explicit budget sentences/vocabulary and stable model IDs, and extend `src/lm/statistical/ngram.py` with allowed target IDs. Do not reimplement n-gram fitting or register count models as `nn.Module` instances.

### Phase 4: Automated, Reproducible Execution

#### 4.1 Suite Entry Point And Run Lifecycle

Requirements: REQ-013, REQ-015. Inputs: current `src/run.py`, collision-prone output naming, per-run artifact needs.

- [x] T014 [Plan:4.1] Extend `src/run.py` with an opt-in suite entry point and stages `preflight`, `prepare`, `train`, `seal`, `evaluate`, `report`, and `all`; retain the current single-config command and expose corpus filters, dry-run, and resume.
- [ ] T015 [Plan:4.1] Implement per-run IDs and atomic state/result files in `src/lm/experiments.py`, fingerprinting resolved model/config, corpus/budget/vocab/target-support hashes, source code including dirty changes, and execution environment.
- [ ] T016 [Plan:4.1] Implement verified resumption in `src/lm/experiments.py`: reuse completed bundles only when fingerprints and required artifacts match; retain failed/interrupted states; restart incomplete fits; never delete or overwrite historical mismatches automatically.

#### 4.2 Selection, Scheduling, And Test Freeze

Requirements: REQ-003, REQ-008, REQ-009, REQ-013, REQ-015. Inputs: Direction 1 selection protocol and single-device resource constraints.

- [ ] T017 [Plan:4.2] Implement fixed-grid validation-only tuning per corpus/budget in `src/lm/experiments.py`, storing all trials, deterministic tie-breaking, selected settings, tuning seed, cost, and final three-seed cold-start fits.
- [ ] T018 [Plan:4.2] Add serial default scheduling, explicit resource checks, synchronized timing, and epochs/updates/tokens-processed instrumentation in `src/lm/experiments.py` and `src/lm/neural/trainer.py`; resource failures remain visible rather than changing the task silently.
- [ ] T019 [Plan:4.2] Separate the existing controller's train/validation/save operations from its immediate test evaluation through a backward-compatible suite path, then seal the matrix, data/vocabulary hashes, diagnostic rules, and validation-selected settings before the first final test score. Verify them in `src/lm/experiments.py` before evaluation/resume; `all` seals automatically after validation selection, while staged operation permits review.

### Phase 5: Diagnostics, Tables, And Graphs

#### 5.1 Frequency And Sequence Analysis

Requirements: REQ-011. Inputs: original OOV flags, shared event IDs, training-only count stores, fixed analysis rules.

- [ ] T020 [Plan:5.1] Add diagnostics in `src/lm/evaluation.py`: current-budget mapped-frequency bins, separate smallest-budget original-word reference groups, budget-specific known-word/OOV states and transitions, group sizes/NLL/zero rates, and matched unsmoothed-versus-add-k trigram comparisons at every budget.
- [ ] T021 [Plan:5.1] Add reproducible category-based selection of at least ten held-out cases per eligible corpus in `src/lm/reporting.py`; retain original/full prefix, target, mapped form, counts at every budget, all model probabilities, and counterexamples without selecting only favorable outcomes.

#### 5.2 Result Schema And Offline Reporting

Requirements: REQ-010, REQ-012. Inputs: completed run bundles and diagnostic accumulators.

- [ ] T022 [Plan:5.2] Extend the controller's existing run dictionaries, CSVs, histories and pandas seed summaries in `src/lm/reporting.py` rather than rebuilding them: add statistical rows, data/seed/trial IDs, vocabulary policy/source/hash/support/size, PPL/NLL/OOV/top-1/coverage/zero metrics, parameters versus count-table entries, training/scoring/tuning costs, versioned JSON and explicit missing/infinite/overflow statuses.
- [ ] T023 [Plan:5.2] Add Matplotlib charts and an offline HTML/Markdown report in `src/lm/reporting.py`: per-corpus end-to-end PPL versus actual training tokens, accompanying vocabulary-size/OOV curves and changing-support caveats, neural mean/std, runtime/throughput, cost-quality tradeoffs, zero-event rates, frequency losses, smoothing ablation, eligibility, and case tables.

### Phase 6: Verification And Rollout

#### 6.1 Regression Checks And Pilot

Requirements: REQ-016. Inputs: protocol invariants and observed source/runtime discrepancies.

- [ ] T024 [Plan:6.1] Add `src/tests/test_preparation.py` for read-only SQLite-shelve record/schema parity, incompatible-backend and missing-database blocking, checksum verification, document/split disjointness, aliases, nested budgets/vocabulary sets, deterministic budget-specific IDs, shared within-budget hashes, held-out/future-word exclusion, minimum-count vocabulary promotion and nonincreasing OOV, vocabulary-keyed cache invalidation, and preservation of raw source/test hashes.
- [ ] T025 [Plan:6.1] Add `src/tests/test_evaluation.py` for target alignment, normalization on allowed IDs, MLE zero events, EOS/padding counts, per-token versus aggregate NLL, reader parity, and matched-weight neural scoring parity.
- [ ] T026 [Plan:6.1] Add `src/tests/test_experiments.py` for tiny all-family runs, validation-only selection, no warm-starting, interrupted/completed resumption, changed-fingerprint rejection, strict JSON, and honest report handling of missing/infinite results.
- [ ] T027 [Plan:6.1] Run a separately named Inaugural or synthetic pilot through preparation, all four model families, and validation reporting; use a pilot holdout from training/validation rather than inspecting final test results to debug settings. Measure resources before large-corpus execution.

#### 6.2 Documentation And Production Gate

Requirements: REQ-001, REQ-012, REQ-015, REQ-016. Inputs: verified implemented commands, pilot evidence, corpus eligibility decisions.

- [ ] T028 [Plan:6.2] Update `src/README.md` and `src/data-pipeline/README.md` to show the real layout, prepared-data mode, corpus limitations, suite commands, immutable raw splits and per-budget vocabularies, changing-support interpretation, output schema, and legacy versus Direction 1 behavior.
- [ ] T029 [Plan:6.2] Update `src/requirements.txt` with tested reporting/audit dependencies and capture exact dependency versions, interpreter, hardware, source/access/licence metadata, and AI/adapted-code declarations in every experiment manifest.
- [ ] T030 [Plan:6.2] Freeze the primary corpus and eligible replication list before final testing, verify acceptance criteria, execute the approved matrix, and retain blocked/failed results and counterexamples in the final report; do not claim universal crossover or all-seven completion without evidence.

## Proposed Project Structure

Reuse existing package boundaries; these are proposed files, not additional frameworks.

```text
src/
  run.py                         # legacy command plus opt-in suite mode
  configs/data_efficiency.yaml    # explicit corpus/model/budget policy
  lm/
    data.py                      # shared reader, per-budget vocab, indexed inputs
    preparation.py               # provenance, partition and budget manifests
    experiments.py               # matrix, tuning, adapters, lifecycle
    evaluation.py                # event scoring and diagnostics
    reporting.py                 # tables, charts, static report
    neural/                      # canonical existing models/trainer
    statistical/
      ngram.py                   # existing model, explicit prediction support
      counts.py                  # conditional: only if measured count storage needs it
  tests/
    test_preparation.py
    test_evaluation.py
    test_experiments.py
```

Do not create a new neural framework, duplicate the old CLI, or place statistical models in the neural registry. Split a proposed module further only when implementation complexity requires it.

## Proposed Configuration Shape

This is a design example, not a config accepted by today's runner. All seven entries point to the imported layout; missing provenance and reader compatibility still block execution.

```yaml
schema_version: 1
name: data_efficiency_per_budget_v1
output_root: results/data_efficiency
primary_corpus: movie-reviews
budgets_percent: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
data_seed: 42
budget_seed: 55
validation_fraction_of_existing_train: 0.10
vocab: {mode: per_budget, source: current_budget_train, min_freq: 2}
preprocessing: {tokenizer: whitespace_v1, lowercase: true}
prepared_data:
  root: data/processed
  import_manifest: data/processed/corpus_import_manifest.json
  rebuild: false
  document_reader: sqlite_shelve_readonly
corpora:
  blog-posts:
    train: data/processed/blog-posts/train.txt
    test: data/processed/blog-posts/test.txt
    database: data/processed/blog-posts/blog-posts.db
    legacy_rare_report: data/processed/blog-posts/rare-ngrams.csv
    grouping: source_file
  gutenberg:
    train: data/processed/gutenberg/train.txt
    test: data/processed/gutenberg/test.txt
    database: data/processed/gutenberg/gutenberg.db
    legacy_rare_report: data/processed/gutenberg/rare-ngrams.csv
    grouping: source_file
  inaugural:
    train: data/processed/inaugural/train.txt
    test: data/processed/inaugural/test.txt
    database: data/processed/inaugural/inaugural.db
    legacy_rare_report: data/processed/inaugural/rare-ngrams.csv
    grouping: source_file
  movie-reviews:
    train: data/processed/movie-reviews/train.txt
    test: data/processed/movie-reviews/test.txt
    database: data/processed/movie-reviews/movie-reviews.db
    legacy_rare_report: data/processed/movie-reviews/rare-ngrams.csv
    grouping: source_file
  reuters:
    train: data/processed/reuters/train.txt
    test: data/processed/reuters/test.txt
    database: data/processed/reuters/reuters.db
    legacy_rare_report: data/processed/reuters/rare-ngrams.csv
    grouping: source_file
  rocstory:
    train: data/processed/rocstory/train.txt
    test: data/processed/rocstory/test.txt
    database: null
    legacy_rare_report: data/processed/rocstory/rare-ngrams.csv
    grouping: story_id
    eligibility: needs_provenance
  webtext:
    train: data/processed/webtext/train.txt
    test: data/processed/webtext/test.txt
    database: data/processed/webtext/webtext.db
    legacy_rare_report: data/processed/webtext/rare-ngrams.csv
    grouping: variant_group
    eligibility: needs_repair
models:
  - {id: bigram_mle, family: ngram, order: 2, smoothing: mle}
  - {id: trigram_mle, family: ngram, order: 3, smoothing: mle}
  - id: trigram_addk
    family: ngram
    order: 3
    smoothing: laplace
    tuning: {k: [0.01, 0.1, 1.0]}
  - id: lstm
    family: neural
    architecture: {embed_dim: 128, hidden_dim: 256, num_layers: 1, dropout: 0.3}
    tuning: {lr: [0.0003, 0.001, 0.003], seed: 7}
    final_seeds: [13, 42, 2026]
execution:
  neural_workers: 1
  resume_completed: true
  incomplete_fit_policy: restart
  ineligible_corpus_policy: record_and_skip
analysis:
  minimum_cases_per_corpus: 10
  frequency_reference: current_budget_mapped
  fixed_reference: smallest_budget_original_words
```

Resource limits must be explicit and calibrated by the pilot, not invented as guarantees. Optional FFNN and alternative smoothers require distinct model IDs/configs. A primary corpus blocked by preflight is a visible failure of the primary study, not permission to silently substitute another corpus.

## Proposed User Workflow

Commands below are future interfaces. They do not work until the planned entry point is implemented. Run from the repository root with the selected experiment environment.

```sh
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency.yaml --stage preflight --dry-run
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency.yaml --stage prepare
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency.yaml --stage train
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency.yaml --stage seal
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency.yaml --stage evaluate --resume
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency.yaml --stage report
```

After protocol approval, the automated equivalent is one `--stage all --resume` invocation. Use `--corpora inaugural` for a separately identified pilot, never as an unrecorded reduction of the production matrix.

Execution order is: inspect -> prepare/freeze -> tune/train on validation -> seal -> score final tests -> aggregate/report. Budget fits can be scheduled independently after preparation because their document/vocabulary manifests are already fixed.

Run state is `pending`, `running`, `completed`, `failed`, `interrupted`, or `blocked`. A completed infinite-PPL result remains completed. Atomic artifact commits, not the existence of an empty `.done` file, determine completion.

## Output And Reporting Contract

```text
results/data_efficiency/<experiment_fingerprint>/
  manifest.json                  # protocol, sources, environment, code
  preflight.json                 # all seven corpora and eligibility
  <corpus>/
    prepared/                    # document/offset/budget manifests, per-budget vocab
    trials/                      # validation-only candidate histories/selection
    runs/<budget>/<model>/<seed-or-deterministic>/<run_hash>/
      state.json
      config.yaml
      metrics.json
      history.csv                # neural runs
      model.pt                   # neural weights with linked vocab/config
    runs.csv
    strata.csv
    cases.jsonl
    figures/
    report.html
  summary.csv
  report.html
```

Every metric identifies the evaluation split, budget-specific vocabulary policy/source/hash/support/size, and prediction-event count. Record content-word tokens separately from scored tokens, which also include EOS. Separate trainable neural parameters, statistical count entries, and serialized bytes; these are not interchangeable parameter counts. Vocabulary size and validation/test OOV rates are shared within a budget and measured anew at each budget.

Use strict JSON: represent infinite PPL as `value: null` plus `status: infinite`, with zero-event counts retained. Distinguish finite-value overflow from genuine zero-probability infinity and from missing/failed runs. CSV may display `inf` alongside an explicit status column.

Required graphs are per corpus:

- Test PPL versus actual training tokens, showing neural mean and sample standard deviation across final seeds. Label it an end-to-end data/vocabulary-growth curve, not a fixed-target comparison. Mark or panel infinite MLE results; never replace infinity with a fabricated large value.
- Vocabulary size and validation/test OOV rate versus training tokens, adjacent to the PPL curves. Show unchanged values honestly; vocabulary need not grow at every step.
- Zero-probability event rate versus training tokens, so unsmoothed learning remains visible when PPL is infinite.
- Fit and scoring cost/throughput versus training tokens, plus a quality-versus-cost plot. Include tuning cost separately.
- Frequency-conditioned NLL/zero-event curves with group counts and fixed versus current-budget definitions clearly distinguished.
- Matched trigram smoothing ablation and at least ten aligned qualitative cases across budgets/models.

Aggregate statistical models as deterministic single runs, not fake three-seed variance. Do not label neural seed standard deviations as confidence intervals over documents. PPL-of-mean-NLL and mean-of-seed-PPL are different quantities; name any reported aggregate explicitly.

Do not average or rank raw perplexities across corpora with different vocabularies, or interpret cross-budget changes as an identical-target intervention. Cross-corpus views should show separate panels and within-corpus, within-budget model comparisons, along with vocabulary/OOV context, eligibility, and missing-data coverage. An absent crossover is a valid outcome.

## Verification Strategy And Acceptance Criteria

This is a Python batch CLI, not a service/browser application. Use focused tests with tiny synthetic corpora and CPU models, then a measured pilot with the explicit experiment interpreter. No Docker, web server, or browser automation is required for the core suite; manually inspect generated report/graphs after their consistency checks.

Acceptance gates:

1. Preparation resolves the explicit imported `data/processed` paths, verifies import and DB checksums, proves its document-reader/backend compatibility, preserves source/test hashes, proves document membership/disjointness, and records duplicate decisions. It cannot rebuild/re-split imported sources, silently replace a missing ROCStories store, or convert missing provenance into sentence budgets.
2. Every budget membership and lexical vocabulary set is nested; rounding aliases are visible. Each vocabulary is derived only from its budget's training subset under the same cutoff, with identical vocabulary/ID/support hashes across models/trials/seeds at that budget. Raw validation/test event IDs and counts stay identical across budgets; mapped IDs may differ and OOV must not increase. A synthetic word crossing the cutoff becomes lexical without admitting future-only or held-out-only words; mapped-count/encoding caches cannot cross vocabulary hashes.
3. All four required families at a budget run on the same mapped targets; an L-token sentence produces L+1 events; padding/BOS are excluded correctly. Neural batch and single-sentence scoring agree within a declared numeric tolerance, including the standalone parity fixture.
4. Add-k normalizes on the allowed target set for seen and unseen histories. MLE zeros yield honest infinity and zero-event counts, not exceptions or clipped finite scores.
5. No final test metrics influence tuning, stopping, example-category definitions, primary-corpus selection, or later changes in the same experiment ID.
6. At least three independent final neural seeds are present per completed production condition, or a deviation is explicitly approved and labeled. Statistical fits are not repeated merely to create variance.
7. Resume skips only fully verified matching artifacts. Changed data/config/code/environment/support fingerprints cannot overwrite or masquerade as old results; interrupted fits restart cleanly.
8. Large-corpus indexing, count storage, sentence-length limits, and scoring pass measured memory/disk/device gates. Unsupported corpus/budget conditions remain explicit failures/blocks.
9. Reported metrics reproduce aggregate tables; graphs retain missing/infinite statuses and correct token counts. Reports contain the smoothing ablation and at least ten reproducible cases per eligible corpus, including unfavorable examples.
10. Legacy commands retain documented behavior. A synthetic end-to-end run and a separately named validation-only pilot pass before sealing the approved production study.

Pilot success means protocol and resource correctness, not a particular PPL, neural advantage, or crossover. Corpus eligibility and research distinctness remain prerequisites for the final matrix.

## Requirement Mapping

Implementation evidence below names planned completion artifacts, not code already written.

| REQ ID | Description | Plan Items | Implementation Evidence |
|---|---|---|---|
| REQ-001 | Seven-corpus intake and eligibility | 1.1, 6.2 | Suite YAML, preflight report, documented eligible replication list |
| REQ-002 | Document provenance and frozen partitions | 1.2 | Preparation manifests, overlap/duplicate audit, preserved test hashes |
| REQ-003 | Nested budgets and independent fits | 2.1, 4.2 | Budget document lists, model histories, no-warm-start checks |
| REQ-004 | Training-only vocabulary per budget | 2.1 | Per-budget vocab/source/support hashes, within-budget equality, cutoff-promotion/OOV and leakage checks |
| REQ-005 | Required and optional model families | 3.2 | Statistical/neural dispatch and declared model IDs |
| REQ-006 | Canonical neural implementation | 3.2 | Modular trainer integration and adapted scorer parity checks |
| REQ-007 | Shared event scoring and target support | 3.1, 3.2 | Shared reader/scorer, target IDs and normalization tests |
| REQ-008 | Validation-only selection | 4.2 | Trial ledger, selected settings and sealed protocol |
| REQ-009 | Seeds and execution conditions | 4.2 | Run metadata, seed separation and synchronized timing |
| REQ-010 | Correct metrics and accounting | 3.1, 5.2 | Event accumulators, versioned metrics JSON and aggregate tables |
| REQ-011 | Diagnostics, ablation and cases | 5.1 | Strata tables, matched trigram comparison and cases JSONL |
| REQ-012 | Graphs and offline reporting | 5.2, 6.2 | Per-corpus figures/reports and suite coverage report |
| REQ-013 | Automation and verified resumption | 1.1, 4.1, 4.2 | CLI stages, matrix, state transitions and atomic run bundles |
| REQ-014 | Large-corpus resource handling | 2.2 | Indexed data, count-store backend and measured resource checks |
| REQ-015 | Reproducibility and provenance | 1.1, 1.2, 4.1, 4.2, 6.2 | Source/config/code/environment hashes, immutable manifests and dependency records |
| REQ-016 | Validation, docs and rollout gates | 6.1, 6.2 | Focused tests, pilot evidence, updated READMEs and production readiness decision |

## Recommended Implementation Order

Complete Phases 1-3 and their invariant tests first. This yields one verified corpus/budget/all-model comparison without hiding data/scoring errors inside a large sweep. Next add Phase 4 automation and Phase 5 reports. Finally run the pilot/resource gates and approve the production matrix.

Keep the first integration thin: use one provenance-verified corpus, existing model classes/trainer/count adapter, one shared result-returning evaluator, and the established CSV outputs. Add the ten-budget loop and plots around that working path. Add large-corpus storage machinery only where the measured pilot requires it; do not build another training framework, another n-gram implementation, or a competing CLI.

Do not start a seven-corpus training sweep with the current percentage flag and add graphs afterward. Per-budget vocabulary is intentional, but sentence-prefix sampling, unresolved document grouping, and inconsistent scoring paths still violate the planned experiment.