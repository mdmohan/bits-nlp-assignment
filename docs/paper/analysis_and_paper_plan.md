# PS-1 analysis, submission layout and research recommendations (six-corpus V3)

Updated 6 November 2026 for Group 1. This report now includes Reuters, ROCStories and WebText alongside Brown, Gutenberg and Inaugural. The editable manuscript is [paper_working_draft.md](paper_working_draft.md), version 0.4. Section 8 ranks further research using these results; recommendations are not completed experiments or authorization to train.

Saved metrics come from `results/comparison_sweep_20261102_v3/full/runs`. Fresh six-corpus summaries and derived event analyses are in [../../results/paper_v3_six_corpora_20261106](../../results/paper_v3_six_corpora_20261106). Earlier three-corpus outputs remain intact. The combined [audit](../../results/paper_v3_six_corpora_20261106/reanalysis_checks.json) covers 390 test conditions: 330 finite PPLs agree with saved event probabilities within relative error below 3.8e-7; 60 reproduce infinite MLE PPL. The original 225-condition audit is retained, and 165 added conditions were newly reconstructed. No training or checkpoint inference was performed for this update.

## Submission Readiness: Step 2 (7 November 2026)

The [shared Google Doc](https://docs.google.com/document/d/1sqjul74OP2pDjQJNfArKm_zkbpQm0Q7XzrsSBHQcemY/edit) is the manuscript editing and review location. Step 1 saved the named version `Submission Baseline - v0.4 - 2026-11-07` and enabled Suggesting mode. The local manuscript remains unchanged; its replacement abstract is pending. This section records the Step 2 inventory, not approval of the final submission.

### Evidence Inventory

Current filesystem and CSV checks confirm the following artifacts under [the canonical V3 runs](../../results/comparison_sweep_20261102_v3/full/runs). Each corpus has exactly five budgets: 20%, 40%, 60%, 80% and 100%. Test conditions are unique corpus/budget/model/seed keys, not independent document replicates.

| Corpus / run directory | Neural seeds | Final checkpoints / histories | Test conditions | Comparison PNGs | Serialized count tables |
|---|---|---:|---:|---:|---:|
| Brown / `brown_press_neural` | 13, 42, 2026 | 45 / 45 | 75 | 6 | 5 |
| Gutenberg / `gutenberg_neural` | 13, 42, 2026 | 45 / 45 | 75 | 6 | 0 |
| Inaugural / `inaugural_neural` | 13, 42, 2026 | 45 / 45 | 75 | 6 | 0 |
| Reuters / `reuters_neural` | 13, 42, 2026 | 45 / 45 | 75 | 6 | 5 |
| ROCStories / `roc-stories` | 13 | 15 / 15 | 45 | 6 | 5 |
| WebText / `webtext-nn` | 13 | 15 / 15 | 45 | 6 | 5 |
| Total | Four three-seed corpora; two single-seed corpora | 210 / 210 | 390 | 36 | 20 |

All nine models occur at every budget with their expected seeds: six deterministic bigram/trigram MLE, Laplace and backoff variants, FFNN context 1, FFNN context 2 and LSTM. Statistical seed fields are blank. All six runs retain nonempty scoring-event archives, artifact manifests and saved passing event-verification receipts. The [combined reanalysis audit](../../results/paper_v3_six_corpora_20261106/reanalysis_checks.json) covers 330 finite and 60 infinite test PPLs. Step 2 checked current metadata and CSV keys; it did not repeat event reconstruction, load checkpoints or freshly hash every artifact.

The [six-corpus paper tables](../../results/paper_v3_six_corpora_20261106/tables) contain 36 CSV/Markdown files, including 270 seed-aggregated test rows. Its [figure directory](../../results/paper_v3_six_corpora_20261106/figures) contains eight PNGs. Brown, Gutenberg and Inaugural event panels remain in [the earlier recheck figures](../../results/paper_v3_recheck_20261105/figures) and must be included when packaging. The [Inaugural Torch-summary set](../../results/inaugural_100pct_torch_summaries) retains three TXT, three HTML and three PNG outputs, with [checkpoint verification metadata](../../results/inaugural_100pct_torch_summaries/verification.json): 100% budget, seed 13, not six independent architecture examples.

The frozen V3 [checksum sidecar](../../colab/comparison_sweep_frozen_v3.zip.sha256) records SHA256 `2ae43095c6e7f0fc95468b13a40c9d49b9004c7579bb5b961b2f7f6e60bbfff8`. Frozen configurations, training snapshots and recorded runtime adaptations remain the experiment authority. Saved [recovery](../../results/comparison_sweep_20261102_v3/comparison_attempts/kubeflow_recovery_20261103/local_verification.json) and [remaining-corpus import](../../results/comparison_sweep_20261102_v3/comparison_attempts/kubeflow_remaining_20261105/local_verification.json) receipts document prior hash checks; they are not a fresh Step 2 integrity certification.

Two preservation gaps remain: ten manifest-listed count tables are absent locally, covering all Gutenberg/Inaugural budgets; a separate Gutenberg resume checkpoint is zero bytes and is not counted among the 210 usable final checkpoints. Do not delete, refit or substitute artifacts to hide these gaps. Saved events still support the planned accuracy analysis, but complete local serialized count-model replay is unavailable. Original Kubeflow environment records identify Python 3.11.11, Torch 2.11.0+cu126 and A100 80GB. Brown signatures record Torch 2.11.0+cu130, but a complete original Colab environment record was not found locally. Supplemental Torch summaries used Torch 2.14.1; do not use that environment as the training lock.

### Requirement Mapping

The three-page [original rubric](../AIMLCZG530_NLP_Group_Assignment_20_Marks_30_Days.pdf) was reread for this inventory. "Present" below means evidence exists, not that all integrity or submission gates have passed.

| Requirement | Current evidence / status | Remaining work or decision |
|---|---|---|
| PS-1 corpus and distinct group scope | Brown is an explicit option; five supplementary public corpora are available | Confirm supplementary grading scope and distinctness from Groups 6/11 |
| Baseline, neural comparator and ablation | Nine variants and smoothing ablation present | User confirmed the existing sentence-prefix LSTM; report that trained model exactly, with no four-word replacement experiment |
| PPL, vocabulary, OOV, runtime and comparisons | Saved six-corpus tables, events and plots present | Extend accuracy by target/history strata in Step 3; preserve denominators and infinite MLE results |
| At least ten analysed sequences and five scholarly references | Fifteen case interpretations and seventeen references drafted | Retain individual interpretations; finish literature and citation review |
| Multiple seeds where computationally feasible | Three seeds on four corpora; one on ROCStories/WebText | Record a genuine single-run justification or authorize additional experiments; no invented SD |
| No train-test sentence/document/template/near-duplicate overlap | Exact audits report zero normalized sentence overlaps | Resolve retained Gutenberg/Reuters flags and missing ROCStories near scan; see per-corpus matrix below |
| Proposal page with research question, data, models, metrics, ablation, risks and roles | Outline exists in this plan | Finish standalone proposal and role allocation |
| Dataset card and obtainable data/access instructions | Source, split and audit metadata partly available | Consolidate versions, URLs/access dates, sizes/tags, rights, preprocessing and limitations; verify upstream ROCStories rights |
| Reproducible environment and README invocations | Requirements, source, notebooks and runtime records partly available | Separate final V3 from pilot commands, document side effects and unknown hardware facts, recover original versions where possible |
| Contributions and AI/tool declaration | Three named authors and an unreviewed AI-use note | Confirm approved roster against four-person rubric header; complete reviewed member/tool declarations |
| 4,000-5,500-word main paper, excluding references/appendices | Working Markdown and DOCX present; draft is overlength | Separate Methodology/Experimental Setup, create real appendices, remove work log, check final word count and export PDF |
| One group submission archive with source/notebooks, evidence, figures and access information | Required experimental material largely present | Finish documents, explicitly include ignored artifacts, verify archive/links/access and portal limits |

### Corpus Integrity and Decisions

All six exact audits report zero lowercase/whitespace-normalized cross-split sentence overlaps. Near-audit screening uses at least ten tokens, at least five distinct five-token shingles and Jaccard similarity at least 0.9; a negative screen is not proof of absence of shorter, lower-similarity or semantic overlap.

| Corpus | Near-audit evidence | Submission disposition |
|---|---|---|
| Brown | No candidates above threshold | Explicit PS-1 corpus; audit-supported within the recorded scope |
| Gutenberg | Two retained train-test pairs | Integrity decision open; audit-only acceptance is not an instructor waiver |
| Inaugural | No candidates above threshold | Audit-supported within scope; supplementary grading scope open |
| Reuters | Six flagged pairs: three train-test involving two distinct test sentences, two train-validation, one validation-test | Integrity decision open; retain the original flagged-pair evidence |
| ROCStories | Near-duplicate scan not run | Near-overlap compliance unknown; supplementary scope and single-seed justification open |
| WebText | No candidates above threshold; one held-out source | Audit-supported within scope; supplementary scope and single-seed justification open |

Step 2 is complete as an inventory and requirement mapping. Final submission remains conditional on team/instructor decisions about roster, supplementary scope and unresolved overlap evidence, plus truthful seed/rights declarations. Remediation with matching reruns, exclusion from graded evidence or an instructor disposition must be explicit; disclosure alone is not compliance. No models were trained, fitted, inferred or rescored during this step. Step 3 can proceed on existing saved events without waiting for a final submission decision.

## Saved-Event Accuracy Audit: Step 3 (7 November 2026)

Step 3 is complete for all six corpora, five budgets and nine models: 390 test conditions, comprising 210 neural seed runs and 180 deterministic statistical conditions. The analysis reads saved scoring events, budget vocabularies and training/test text; it does not fit probabilities, load checkpoints, infer or rescore predictions. The existing [analyzer](../../src/analyze_comparison_sweep.py) now provides an opt-in `--accuracy` path. Original results and manuscript prose remain unchanged.

### Outputs and Verification

- [Per-condition accuracy](../../results/paper_accuracy_20261107/tables/accuracy_by_condition.csv): 390 rows with overall, non-UNK-target, in-vocabulary, OOV-label and UNK-target accuracy, their exact denominators, predicted-UNK fractions and separate EOS accounting.
- [Per-condition context accuracy](../../results/paper_accuracy_20261107/tables/accuracy_by_context.csv): 7,020 rows, partitioning seen/unseen histories by overall, in-vocabulary and OOV targets for native, common one-token and common two-token histories.
- [Seed-aware accuracy summary](../../results/paper_accuracy_20261107/tables/accuracy_summary.csv): 270 model/budget/corpus rows; means and sample SD use only the available seeds. Single-run SD is undefined, not zero.
- [Seed-aware context summary](../../results/paper_accuracy_20261107/tables/context_accuracy_summary.csv): 4,860 rows. Empty-group accuracy remains undefined.
- [Execution checks](../../results/paper_accuracy_20261107/accuracy_checks.json): canonical test hashes and occurrence alignment, budget/variant vocabulary agreement, target and prediction IDs, OOV flags, correctness, event weights and reconstructed statistical history coverage.

All 390 conditions exactly reconstruct saved word counts, correct predictions, OOV counts and overall accuracy. Every target/history partition closes in both event counts and correct predictions. Independent checks confirmed keys, seeds, summaries and all nine full-budget WebText models against the prior audit; maximum share-closure error was 1.11e-16. The six focused [analysis-only tests](../../src/tests/test_analysis.py) pass; model-fitting tests were deliberately not run.

### Definitions

Accuracy uses word events only; EOS is reported separately. In-vocabulary targets are raw words retained by that budget's vocabulary. OOV accuracy measures correctness after all omitted words map to `<unk>`; it is not lexical recovery. Excluding mapped UNK targets and selecting in-vocabulary targets are distinct definitions in general, but equal in all 390 conditions here because no literal UNK targets were observed.

History membership uses mapped tokens and includes training histories whose next target is EOS. FFNN context 1 uses one preceding token; context 2 uses two inputs, initially PAD/BOS rather than the statistical trigram's BOS/BOS. Their initial support booleans agree. LSTM native membership means an exact mapped sentence prefix beginning with BOS; this is neither unseen n-gram continuation status nor a claim about hidden-state familiarity. Once a prefix fails to match, it cannot restart within that sentence. Common one/two-token strata permit shared-population comparisons across architectures.

### Full-Budget Findings

Trigram Laplace accuracy below is reported as percentages; the CSVs store fractions. OOV share is budget-specific. The last column counts native-unseen word events, all of which selected UNK; their correct predictions consist only of mapped OOV-label successes.

| Corpus | Overall accuracy | In-vocabulary accuracy | OOV-label accuracy | OOV word share | Native-unseen words / OOV-correct |
|---|---:|---:|---:|---:|---:|
| Brown | 12.77% | 11.04% | 40.98% | 5.80% | 28,978 / 1,417 |
| Gutenberg | 10.33% | 8.63% | 26.99% | 9.28% | 385,350 / 24,515 |
| Inaugural | 12.60% | 10.97% | 39.46% | 5.73% | 4,200 / 247 |
| Reuters | 18.91% | 18.13% | 35.80% | 4.39% | 35,211 / 1,992 |
| ROCStories | 17.37% | 17.39% | 15.30% | 0.90% | 44,139 / 586 |
| WebText | 41.41% | 1.55% | 82.54% | 49.21% | 2,881 / 1,442 |

WebText makes the artifact especially visible: 1,442 of 1,470 full-budget trigram Laplace successes predict UNK; only 28/1,803 in-vocabulary targets are correct. Trigram backoff instead achieves 92/1,803, or 5.10%, in-vocabulary accuracy. All 1,442 Laplace OOV successes occur on unseen two-token histories. Brown differs from the other five full-budget corpora: 1,349 additional OOV successes occur on seen histories, so OOV hits must not universally be attributed to unseen-history tie-breaking.

Overall and in-vocabulary accuracy winner sets differ in 15 of 30 corpus/budget pairs, including five tie-only changes; ten pairs have no common winner. Neural rankings average existing seeds only. These are accuracy rankings, not replacements for PPL rankings. MLE's retained-word top-1 ties with Laplace do not repair its infinite held-out PPL.

```bash
python -B -u src/analyze_comparison_sweep.py --accuracy \
   --corpora brown gutenberg inaugural reuters roc-stories webtext \
   --out results/paper_accuracy_20261107
```

The verified local execution used `/Users/ajoish/.venvs/nlp-lm/bin/python`, outside OneDrive. This command is analysis-only; its full archive scan can take several minutes. Step 4 remains separate: validate and attribute Laplace's uniform unseen-history tie behavior, including any seen-history ties and an explicit UNK baseline, before rewriting the paper. This audit does not resolve the Step 2 corpus-integrity or submission gates.

## Laplace Tie Attribution: Step 4 (7 November 2026)

Step 4 is complete for all 60 Laplace corpus/budget/order conditions. The [verification receipt](../../results/paper_laplace_audit_20261107/laplace_checks.json) records a fresh saved-event scan comparing 60 Laplace conditions with their 60 MLE counterparts. No predictions were rescored, probabilities changed or models fitted. The maximum deviation from the expected unseen-history log probability was 1.78e-15.

The actual [prediction path](../../src/lm/cli.py) selects the lowest token ID among maximum-count followers for an observed history. Add-one does not change this count ordering, and all saved observed-history MLE/Laplace prediction digests agree. An unseen history has no followers: [add-one probability](../../src/lm/statistical/ngram.py) is uniform over the allowed support, and the adapter selects its minimum ID, UNK=1. PAD=0 and BOS=2 are excluded; EOS=3 remains allowed. Thus uniform-tie UNK successes are a decision-rule artifact, not evidence that Laplace recovered unseen spellings. This does not invalidate its finite probabilities or change the existing PPL comparison.

The [attribution table](../../results/paper_laplace_audit_20261107/tables/laplace_tie_attribution.csv) separates observed-history correctness from unseen-history UNK correctness. The [always-UNK baseline](../../results/paper_laplace_audit_20261107/tables/always_unk_baseline.csv) has 30 rows, one per corpus/budget; the [model comparison](../../results/paper_laplace_audit_20261107/tables/accuracy_vs_unk_baseline.csv) has 390 rows using each actual seed. Baseline accuracy is mapped-UNK target count divided by all word events, equalling OOV share in these data. It is an accuracy-only diagnostic, not a separately scored language model.

Let $N$ be all test words, $U$ words with unseen histories, $C_s$ observed-history successes and $C_u$ unseen-history UNK successes. Reported accuracy is $(C_s+C_u)/N$. The contribution attributable to selecting UNK on uniform ties is $C_u/N$. An abstention counterfactual retains the full denominator and has accuracy $C_s/N$. A uniformly random unseen-history tie break has expected accuracy $(C_s+U/K)/N$, with $K$ including EOS and excluding PAD/BOS. These expectations change only the hypothetical selected label; they do not alter saved probabilities, PPL or actual predictions.

| Corpus, 100% trigram Laplace | Always-UNK accuracy | Actual accuracy | Uniform-tie contribution | Abstention accuracy | Random-tie expected accuracy |
|---|---:|---:|---:|---:|---:|
| Brown | 5.80% | 12.77% | 1.22 pp | 11.56% | 11.56% |
| Gutenberg | 9.28% | 10.33% | 2.50 pp | 7.83% | 7.83% |
| Inaugural | 5.73% | 12.60% | 2.26 pp | 10.34% | 10.35% |
| Reuters | 4.39% | 18.91% | 1.57 pp | 17.33% | 17.34% |
| ROCStories | 0.90% | 17.37% | 0.14 pp | 17.23% | 17.24% |
| WebText | 49.21% | 41.41% | 40.62 pp | 0.79% | 0.80% |

At full-budget WebText, 1,442/1,470 successes arise from unseen-history UNK ties; the always-UNK baseline still exceeds Laplace by 7.80 percentage points. Brown's additional 1,349 observed-history UNK successes require a different explanation. Inspection of all five original Brown count archives, guarded by stored-vocabulary and training-snapshot SHA256 equality, partitions the full-budget trigram successes into 1,139 unique count maxima and 210 equal-count maxima resolved by the smaller ID. No observed-history UNK selections occur in the other five corpora at any budget. Missing Gutenberg/Inaugural count archives were not recreated: saved-event agreement and the absence of observed UNK selections suffice for their present attribution, not for an exhaustive count-table replay.

Ten focused fit-free tests and an independent table/receipt verification pass. Count ties mean ties among maximum-frequency continuations, not the minimum ID among all followers. Original prediction code, fitted models and results remain unchanged.

```bash
python -B -u src/analyze_comparison_sweep.py --laplace-audit \
   --corpora brown gutenberg inaugural reuters roc-stories webtext \
   --accuracy-input results/paper_accuracy_20261107 \
   --out results/paper_laplace_audit_20261107
```

## Model Definitions and Context Scope: Step 5 (7 November 2026)

Step 5's source/configuration verification and model-description corrections are complete. The manuscript's count and neural model paragraphs now explicitly describe bigram/trigram histories, context-matched FFNNs, PAD/BOS sentence-start inputs, masked output support, sentence-prefix LSTM state and batches of up to 32 sentences. Existing estimator equations remain unchanged; broader rewriting, architecture screenshots and submission packaging are later steps. No new experiments were authorized or run.

| Model or behavior | Verified implementation | Meaning for the paper |
|---|---|---|
| Bigram/trigram MLE and add-one | [estimators](../../src/lm/statistical/ngram.py), [shared count variants](../../src/model_comparison.py) | One/two-token histories; one counting pass per budget; zero MLE probability on unseen events; add-one over allowed targets |
| Absolute-discount backoff | [backoff implementation](../../src/lm/statistical/ngram.py) | Discount 0.75; lower-order redistribution only for unseen continuations; add-one unigram base; full-support histories use MLE, not discounting |
| FFNN context 1/2 | [FFNN model](../../src/lm/neural/ffnn/model.py) | Embedding 128; concatenate one/two embeddings; tanh hidden 256; dropout 0.3; vocabulary logits; context 2 initially PAD/BOS |
| Sentence-prefix LSTM | [LSTM model](../../src/lm/neural/lstm/model.py), [sentence batching](../../src/lm/data.py) | Embedding 128; one recurrent layer with hidden 256; dropout 0.3 on embeddings/outputs; state resets per sentence |
| Output support and training | [evaluation](../../src/lm/evaluation.py), [trainer](../../src/lm/neural/trainer.py), [saved configuration](../../results/comparison_sweep_20261102_v3/full/runs/inaugural_neural/train_100pct/config.yaml) | PAD/BOS logits masked; Adam 0.001; clip 1.0; maximum 20 epochs; patience 3; strict validation-NLL checkpoint selection |
| Torch-summary dummy shape | [summary generator](../../results/inaugural_100pct_torch_summaries/generate.py) | Input `[32,10]` is an inspection example, not a trained context-window declaration |

For stored vocabulary size $V$ including all four specials, parameter totals are $385V+33{,}024$ for context-1 FFNN, $385V+65{,}792$ for context-2 FFNN and $385V+395{,}264$ for LSTM. The independent review checked these against all 210 saved neural run rows and the 30 configurations. Masking logits does not remove parameters. FFNN context comparisons do not isolate context length from parameter count, and LSTM comparisons also change architecture and effective history length.

On 7 November 2026, the user confirmed that the paper should retain and accurately report the trained sentence-prefix LSTM rather than replace it with a four-word-context model. This supersedes the earlier deferred decision. No four-word training ablation is claimed or required for the agreed scope, and no new training is authorized. Step 5's model descriptions and context-scope decision are complete.

As requested, no further Google Docs comments are added. Necessary model-text changes are made directly and checked against the local snapshot; existing collaborative changes are preserved. The user also authorized replacing the abstract now, ahead of the remaining broader paper rewrite. The approved abstract is used verbatim, including its sentence-level LSTM description; the local Markdown and shared Google Doc are synchronized. No extra abstract claims are added.

### Step 6: Experimental History and Pipeline (7 November 2026)

The manuscript's new Section 3.4 separates exploratory protocols from the final frozen V3 matrix. It preserves the approved abstract and the accepted sentence-prefix LSTM. Import/verification timestamps are not reported as training completion dates, and recovered files are not counted as new fits.

| Stage | Evidence and scope | Relationship to final results |
|---|---|---|
| Corpus-specific experiments and NLTK pilot | [src/configs/inaugural.yaml](../../src/configs/inaugural.yaml), [src/configs/reuters.yaml](../../src/configs/reuters.yaml) and [src/configs/README.md](../../src/configs/README.md) retain earlier configurations. The pilot uses whole-document budgets and `min_freq=2`. | Background development, not pooled with V3. Retained configuration files alone do not date individual executions. |
| Inaugural convergence follow-up | [results/inaugural_convergence_v1/protocol.json](../../results/inaugural_convergence_v1/protocol.json): 10/50/100% budgets, final neural seeds 42/43/44, maximum 20 epochs and patience three. | Separate whole-document protocol, not the final sentence-prefix sweep. |
| Frozen six-corpus comparison | [colab/README.md](../../colab/README.md): V3 bundle dated 2 November; 20/40/60/80/100% sentence prefixes, budget-specific vocabularies, nine model variants. | The completed matrix supplies 210 neural fits and 180 statistical conditions. |
| Artifact recovery and import | [Gutenberg/Inaugural verification](../../results/comparison_sweep_20261102_v3/comparison_attempts/kubeflow_recovery_20261103/local_verification.json) covers 90 neural conditions and explicitly records no training or evaluation during local import. [Remaining-corpus verification](../../results/comparison_sweep_20261102_v3/comparison_attempts/kubeflow_remaining_20261105/local_verification.json) covers 45 Reuters, 15 ROCStories and 15 WebText checkpoints and preservation of the existing runs. | Recovery of existing outputs, not additional experimental conditions. Brown's 45 Colab seed-runs complete the 210-neural-run inventory. |
| Saved-event analysis | Steps 3 and 4 above verify all 390 test conditions and diagnose mapped-UNK accuracy and Laplace ties. | No fitting, checkpoint inference or probability rescoring in these audits. |

The controlling final-run path is `run_full_config` -> `run_from_config` -> `LanguageModelController`: the controller receives the selected training prefix and constructs `Vocab(corpus.train, ...)`. The legacy controller is not evidence of a shared full-budget vocabulary. Frozen prepared manifests bypass download/repartitioning during the run. The whole-document suite in `lm.experiments` is an earlier protocol and must not be used to describe V3 budget allocation.

| Phase | Owning implementation | Input, output and separation rule |
|---|---|---|
| Prepare | [src/prepare_mega_run.py](../../src/prepare_mega_run.py), `freeze_splits` | Document partitions -> train/valid/test, occurrence records and hashes; exact quarantine prioritizes test, then validation. This does not certify absence of all near-duplicates. |
| Budget and vocabulary | [src/run.py](../../src/run.py), `run_full_config`; [src/lm/controller.py](../../src/lm/controller.py), `run_from_config` / controller initialization | Ordered train-sentence prefix -> budget train snapshot and training-only vocabulary; fixed validation/test mapped into that support. Brown threshold two, other corpora one. |
| Neural training | [src/lm/neural/trainer.py](../../src/lm/neural/trainer.py), `Trainer.fit` | Budget train plus fixed validation -> best weights and histories; strict validation NLL selects the checkpoint, not test metrics. |
| Statistical views | [src/model_comparison.py](../../src/model_comparison.py), `make_ngram_variants` | Same budget train/vocabulary -> one trigram counting pass and six shallow-copy estimator views; no seed variation. |
| Base scoring | [src/model_comparison.py](../../src/model_comparison.py), `evaluate_split` / `write_full_event`; [src/lm/evaluation.py](../../src/lm/evaluation.py) | Models plus fixed split text -> metrics and full event archive; EOS contributes to PPL but not word accuracy/OOV. |
| Rare probe | [src/model_comparison.py](../../src/model_comparison.py), `evaluate_rare_file` | Isolated rare n-grams -> frequency-weighted diagnostic events; not pooled into ordinary held-out word accuracy. |
| Reconcile | [src/model_comparison.py](../../src/model_comparison.py), `verify_scoring_events` | Saved event archive plus metrics -> event verification; artifact hashes identify recorded outputs. |
| Offline analysis | [src/analyze_comparison_sweep.py](../../src/analyze_comparison_sweep.py) | Saved metrics/events -> summaries, figures and accuracy audits. Audit modes do not fit models; the comparison CLI is not a read-only substitute because it fits statistical counts. |

Two printable diagrams and an eight-row structured phase map are generated through a documentation-only early-return mode in the existing analyzer. This mode does not open result metrics, event archives or checkpoints. Source paths and phase-table text bounds are checked during rendering.

```bash
/Users/ajoish/.venvs/nlp-lm/bin/python -B src/analyze_comparison_sweep.py --pipeline --out results/paper_methodology_20261107
```

Outputs: [pipeline diagram](../../results/paper_methodology_20261107/figures/experimental_pipeline.png), [phase-to-code diagram](../../results/paper_methodology_20261107/figures/phase_code_reference.png) and [structured phase map](../../results/paper_methodology_20261107/tables/pipeline_phases.csv). Existing aggregate PPL Figure 1 becomes Figure 3 after the two methodology figures. Final PDF/DOCX export and broader manuscript restructuring remain later steps.

Step 6 is complete. Section 3.4 is a real level-three heading in the shared Google Doc, with both PNGs inserted above their respective captions and visually checked at document width. The local Markdown contains the same prose and captions. The saved Google Docs text export is 64,630 characters, SHA256 `937ebf89d97a2f3412ec46fdd7614bac59885a91c43f1a80585ae1cba5bba45e`. Removing only the added subsection and reversing the one existing caption renumbering recovers the entire exact pre-step body. The new subsection's two image placeholders were also checked in order. No comments or suggestions were added; the approved abstract and actual sentence-prefix LSTM remain unchanged.

Focused validation passed: both diagrams are nonblank (1,408 x 1,568 and 1,578 x 1,404 pixels), all new local links resolve, the phase map has eight rows, table labels fit their cells, figure numbering is 1/2/3, and all ten fit-free `AccuracyAnalysisTests` pass. Diagram generation and this documentation update performed no language-model training, checkpoint inference or prediction rescoring.

### Step 7: Actual Torch Summaries and Architecture Illustration (7 November 2026)

The existing [summary receipt](../../results/inaugural_100pct_torch_summaries/verification.json) and [generator](../../results/inaugural_100pct_torch_summaries/generate.py) identify a strict CPU inspection of the saved Inaugural 100%-budget seed-13 checkpoints. The original inspection used PyTorch 2.14.1 and torchinfo 1.8.0, loaded weights strictly and ran dummy forwards. That generator is not rerun for Step 7. Its inspection runtime must not be substituted for the original training environment. The unmodified TXT/HTML/PNG summaries are reused rather than recreating a handwritten model-summary table.

| Model | Embedding parameters | Hidden/recurrent parameters | Output parameters | Total trainable parameters |
|---|---:|---:|---:|---:|
| FFNN context 1 | 1,553,408 | 33,024 | 3,118,952 | 4,705,384 |
| FFNN context 2 | 1,553,408 | 65,792 | 3,118,952 | 4,738,152 |
| Sentence-prefix LSTM | 1,553,408 | 395,264 | 3,118,952 | 5,067,624 |

All use the saved 12,136-entry vocabulary, including four special tokens. Totals match the receipt and all nine saved seed rows. The formulas remain $385V+33{,}024$, $385V+65{,}792$ and $385V+395{,}264$. The LSTM recurrent block comprises 131,072 input weights, 262,144 recurrent weights and two 1,024-entry biases. PAD/BOS masking does not remove output parameters. FFNN context 2 adds 32,768 parameters over context 1, so this is not a capacity-matched context ablation.

The new [architecture diagram](../../results/paper_architecture_20261107/figures/neural_architectures.png) follows [FFNN forward](../../src/lm/neural/ffnn/model.py) and [LSTM forward](../../src/lm/neural/lstm/model.py), with external masking in training/evaluation. It distinguishes causal one/two-token windows from sentence-prefix recurrence and identifies tensor dimensions and source paths. It is an explanatory illustration, not a replacement for the actual summary screenshots.

Appendix A in the manuscript includes all three original screenshots: [FFNN context 1](../../results/inaugural_100pct_torch_summaries/ffnn_ctx1.png), [FFNN context 2](../../results/inaugural_100pct_torch_summaries/ffnn_ctx2.png) and [LSTM](../../results/inaugural_100pct_torch_summaries/lstm.png). The caption and appendix explain that `[32,10]` / `[32,10,12136]` are dummy inspection shapes, not a four- or ten-word LSTM training context. torchinfo memory and mult-add estimates are not measured training runtime or GPU peak memory. The main architecture figure is Figure 3; the previous aggregate PPL figure becomes Figure 4. Appendix figures use A1/A2/A3.

The existing analyzer's new `--architectures` mode verifies receipt totals, the budget vocabulary, all nine saved seed parameter rows and diagram text bounds. It reads saved metadata and summary files only, not checkpoints or scoring archives. It records SHA256 hashes of all nine existing summary TXT/HTML/PNG files in [architecture_checks.json](../../results/paper_architecture_20261107/architecture_checks.json), alongside the [parameter table](../../results/paper_architecture_20261107/tables/neural_architecture_parameters.csv).

```bash
/Users/ajoish/.venvs/nlp-lm/bin/python -B src/analyze_comparison_sweep.py --architectures --out results/paper_architecture_20261107
```

Step 7 is complete. The shared Google Doc contains the architecture diagram in Section 4.2 and all three original Torch-summary PNGs above their respective Appendix A captions. All four images were visually checked in the document; Appendix A is a real level-two heading. The local Markdown has the same prose and captions. The saved text export is 66,330 characters, SHA256 `8a22b0093f14fe36f3e145039e47339d3c0c4060218c0aa6ee3cef2dd65b3056`. Removing only the two added blocks and reversing the PPL caption's Figure 3-to-4 change recovers the exact pre-step manuscript. Each added image placeholder was verified in order. The approved abstract and sentence-prefix LSTM decision remain unchanged, with no comments or suggestions added.

Validation passed: the nine summary TXT/HTML/PNG hashes match the recorded assets, the three parameter totals agree with all nine saved seed rows, the diagram is nonblank and its labels fit, new local links resolve, figure numbering is 1/2/3/4 and A1/A2/A3, and all ten fit-free `AccuracyAnalysisTests` pass. No training, checkpoint loading, dummy forward, prediction rescoring or dependency change was performed for Step 7. Final PDF/DOCX regeneration remains a later submission task.

### Earlier Six-Corpus Analysis Commands

```bash
python src/analyze_comparison_sweep.py --out results/paper_v3_six_corpora_20261106
python src/analyze_comparison_sweep.py --skip-summary --events --out results/paper_v3_six_corpora_20261106
```

The manuscript contains the additional per-seed loss-decomposition recipe. These commands read saved results only; the event pass can take several minutes.

## 1. What the experiment actually is (verified, use in Dataset/Method sections)

| Item | Brown | Gutenberg | Inaugural |
|---|---|---|---|
| Source | Full NLTK Brown, 500 docs, all 15 genres (folder name `brown_press` is misleading) | 18 NLTK Gutenberg books | 60 NLTK inaugural addresses |
| Split (document level, seed 42) | 80/10/10 → 400/50/50 docs | 80/10/10 → 14/2/2 books | 70/20/10 → 42/12/6 addresses |
| Test docs | 50 docs across 14 genres | **Austen *Emma* + KJV Bible** | 1793 Washington, 1817 Monroe, 1849 Taylor, 1857 Buchanan, 1949 Truman, 1977 Carter |
| Valid docs | 50 docs | Chesterton *Ball*, Whitman *Leaves* | 12 addresses |
| Train / test words (100%) | 924,218 / 116,408 | 951,248 / **978,972** | 99,657 / 10,934 |
| Leakage | exact overlap 0; 5,801 quarantined occurrences | exact 0; 624 quarantined; 2 near-dup pairs (Bible quotations, audit-only) | exact 0; 17 quarantined |
| Tokenizer | whitespace, lowercase; Brown punctuation keeps its tag (`,/,` `./.`: 12% of tokens, 33 types) | whitespace; punctuation stays attached (`lord,` `has.`: 16% of tokens) | same as Gutenberg |
| `min_freq` | **2** (`<unk>` is trained) | **1** (`<unk>` never seen in training) | **1** |
| Vocab 20% → 100% | 9,980 → 24,325 | 20,024 → 75,817 | 5,421 → 12,136 |
| Test OOV 20% → 100% | 11.5% → 5.8% | 17.7% → 9.3% | 11.9% → 5.7% |

| Item | Reuters | ROCStories | WebText |
|---|---|---|---|
| Retained documents | 8,212 news documents | 98,161 stories from archived public mirror | 5 source files |
| Train / validation / test documents | 5,748 / 1,643 / 821 | 68,710 / 19,635 / 9,816 | 3 / 1 / 1 |
| Nominal split / seed | 70/20/10; 42 | 70/20/10; 42 | 70/20/10; 3999 |
| Full training / test words | 889,683 / 126,774 | 2,980,263 / 427,651 | 53,269 / 3,550 |
| Test events including EOS | 132,681 | 476,535 | 3,866 |
| `min_freq` | 1 | 1 | 1 |
| Vocab 20% -> 100% | 20,215 -> 55,921 | 28,661 -> 58,271 | 2,107 -> 9,609 |
| Test-word OOV 20% -> 100% | 9.0% -> 4.4% | 2.5% -> 0.9% | 68.5% -> 49.2% |
| Neural seeds | 13, 42, 2026 | 13 only | 13 only |
| Quarantined occurrences | 1,518 | 3,192 | 74 |
| Leakage-audit scope | Zero exact overlaps; six near-duplicate pairs retained as audit-only | Zero exact overlaps; near-duplicate scan not run, explicitly accepted | Zero exact overlaps; no near-duplicate candidates above threshold |

WebText trains on `grail.txt`, `pirates.txt`, `wine.txt`; validates on `overheard.txt` (156,464 words); tests on `singles.txt` (316 sentences). One held-out source cannot establish broad web-domain robustness. ROCStories supplies more text, but the models still reset state each sentence: these are not story-level reasoning scores.

- **Models.** `bigram/trigram_mle` are unsmoothed. `*_laplace` is add-one. `*_backoff` uses absolute discounting (d = 0.75) and backs off recursively to an add-one unigram. `ffnn_ctx1/ctx2` is embed 128 → 256 tanh → softmax with dropout 0.3. `lstm` is 1 layer, embed 128, hidden 256, dropout 0.3.
- **Training.** Adam (lr 1e-3), batch 32, gradient clip 1.0, at most 20 epochs, patience 3 on validation NLL, best checkpoint retained. Three neural seeds on Brown/Gutenberg/Inaugural/Reuters; one on ROCStories/WebText. Total: 210 neural fits plus 180 deterministic statistical conditions from 30 shared count passes. Do not report single-seed SD as zero.
- **Budgets.** 20/40/60/80/100% are prefixes of `train.txt` in sentence order (`budget_unit: sentence_prefix`). Documents are in sorted-ID order, so the budgets are **not random samples**:
  - Brown 20% covers genres A–D plus part of E (it ends inside `ce13`). Fiction genres (K–R) appear only in the 80% and 100% budgets.
  - Inaugural budgets are **chronological**: 20% runs from 1789 to 1861, and post-1989 speeches enter only at 100%.
  - Gutenberg budgets add books alphabetically by author (Austen, Blake, Bryant…; Milton and Shakespeare enter only at 100%).
   - Reuters and ROCStories follow sorted source IDs, not a verified chronology or randomized curriculum. WebText follows source-file order. Its 20% sentence budget contains only 7,745 words, 14.5% of its full training words.
- **Vocabulary.** Rebuilt from each budget's own training text. Prediction support (vocab size) therefore grows with budget.
- **Metrics.** PPL is computed over word tokens plus EOS. Top-1 accuracy is over word tokens. Runtime is fit/train seconds and test-scoring seconds.
- **Hardware.** Brown ran on Colab; the other corpora include Kubeflow training/recovery. Timings are not hardware-normalized, and even within-corpus budget comparisons may reflect different recovery environments.

## 2. Key findings

**Full-budget results** ([generated table](../../results/paper_v3_six_corpora_20261106/tables/main_results_100pct.md)):

| Corpus | Best model (test PPL) | Best backoff | LSTM (mean ± sd where three seeds exist) | Add-one trigram | MLE |
|---|---|---|---|---|---|
| Brown | **LSTM 206.8** | 260.0 (bigram) | 206.8 ± 0.9 (−20.5%) | 9,641 | ∞ (bigram 24.7%, trigram 63.3% zero-prob) |
| Gutenberg | **bigram backoff 914.5** | 914.5 | 1,053.9 ± 2.4 (+15.2%) | 52,181 | ∞ (41.5% / 79.6%) |
| Inaugural | **trigram backoff 342.0** | 342.0 | 387.1 ± 17.4 (+13.2%) | 8,130 | ∞ (41.0% / 76.5%) |
| Reuters | **trigram backoff 291.2** | 291.2 | 308.1 ± 1.0 (+5.8%) | 23,001 | ∞ |
| ROCStories (one seed) | **LSTM 78.2** | 110.4 (trigram) | 78.2 (−29.2%) | 6,477 | ∞ |
| WebText (one seed) | **bigram backoff 7,810.6** | 7,810.6 | 13,526.2 (+73.2%) | 9,381 | ∞ |

1. **Smoothing ablation** (`smoothing_ablation.md`, `figures/sparsity_vs_budget.png`).
   - All 60 MLE test conditions are infinite. Zero-probability coverage remains essential alongside PPL.
   - Add-one usually performs much worse than backoff, but WebText is an exception to the size of that gap: bigram add-one is only 3.4% worse than bigram backoff and beats all full-budget neural models.
   - Growing support helps explain poor add-one scaling, without isolating it from changes in counts and domains. Gutenberg trigram add-one rises from 16,784 to 52,181.
   - Argmax accuracy can change dramatically. WebText trigram add-one reaches 41.41% mapped accuracy, but 1,442 of its 1,470 correct predictions are unknown symbols; retained-word accuracy is only 1.55%. Trigram backoff achieves 5.10% retained-word accuracy. See the [accuracy audit](../../results/paper_v3_six_corpora_20261106/tables/webtext_accuracy_decomposition.csv).
2. **Neural vs backoff.** LSTM wins full-test PPL on Brown and ROCStories; backoff wins on Gutenberg, Inaugural, Reuters and WebText. ROCStories FFNN context 2 also beats backoff (106.8 vs 110.4). WebText's best full-budget neural model is FFNN context 1 (12,211.9), not LSTM. Single-seed winners remain exploratory.
3. **Conditional gains and aggregate losses.** Use one fixed aggregate-best backoff per corpus: bigram on Brown/Gutenberg/WebText, trigram on Inaugural/Reuters/ROCStories. The manuscript gives full category tables. This additive summary uses nats per total test event; negative favors LSTM:

   | Corpus | Non-OOV contribution | OOV contribution | Total LSTM-minus-backoff NLL | LSTM PPL reduction on unseen bigrams with retained targets |
   |---|---:|---:|---:|---:|
   | Brown | -0.23079 | +0.00192 | -0.22887 | 37.2% |
   | Gutenberg | -0.04899 | +0.19093 | +0.14194 | 43.0% |
   | Inaugural | -0.05402 | +0.17724 | +0.12323 | 46.8% |
   | Reuters | -0.13710 | +0.19332 | +0.05622 | 67.3% |
   | ROCStories | -0.40652 | +0.06182 | -0.34469 | 82.3% |
   | WebText | +0.12492 | +0.42423 | +0.54915 | 11.5% |

   - Reuters extends the Gutenberg/Inaugural pattern: OOV losses outweigh non-OOV gains. All five `min_freq=1` corpora lack positive unknown-target examples, but their output parameters still receive softmax updates.
   - ROCStories wins despite conditional OOV PPL around 48 billion: only 0.8% of its scored events are OOV. It also beats trigram backoff on seen trigrams (24.3 vs 26.9), refuting a universal count-model advantage on memorized contexts.
   - WebText loses on non-OOV events too. Its OOV share is 45.2% of scored events, distinct from the 49.2% word-only OOV rate. Dropping OOV would not reverse its LSTM/backoff ranking.
   - All six show an LSTM advantage on unseen mapped bigrams with retained targets. This does not prove rare-word mastery, all-known histories, or a causal benefit from any specific architectural feature.
4. **Fixed cohort.** Keep target occurrences that are non-OOV at 20%, including EOS. This fixes membership, not output support, context mapping, model capacity or data composition. Coverage ranges from 37.1% of WebText events to 97.7% of ROCStories events:

   | Corpus | Ratio at 20% | Ratio at 100% |
   |---|---|---|
   | Brown | 0.848 | 0.758 |
   | Gutenberg | 1.184 | **0.953** (LSTM 219.1 ± 13.1 vs 229.9; crossover between 80% and 100%) |
   | Inaugural | 0.983 | 0.943 |
   | Reuters | 0.986 | 0.836 |
   | ROCStories | 0.744 | 0.648 |
   | WebText | 1.039 | 1.215 |

   Both backoff orders and LSTM improve monotonically in five corpora. WebText LSTM instead rises from 256.1 to 277.3, and its backoff curves have a final-step reversal. Add-one generally worsens. Brown aggregate LSTM PPL rises (164.5 -> 206.8) while cohort PPL falls; support growth is a plausible explanation, not an isolated causal estimate.
5. **Aggregate data-efficiency trend** (`neural_vs_backoff_ratio.png`): best neural over best backoff.

   | Corpus | 20% | 40% | 60% | 80% | 100% |
   |---|---|---|---|---|---|
   | Brown | 0.859 | 0.833 | 0.808 | 0.800 | 0.795 |
   | Gutenberg | 1.201 | 1.182 | 1.133 | 1.129 | 1.153 |
   | Inaugural | 1.323 | 1.220 | 1.186 | 1.146 | 1.132 |
   | Reuters | 1.298 | 1.181 | 1.118 | 1.108 | 1.058 |
   | ROCStories | 0.832 | 0.776 | 0.728 | 0.710 | 0.708 |
   | WebText | 0.738 | 1.670 | 1.405 | 1.266 | 1.564 |

   Endpoint improvement holds in five corpora, not WebText. Gutenberg worsens between 80% and 100%. WebText's best-neural curve switches architectures and reverses from a neural win at 20% to backoff wins thereafter; it is not a fixed-model learning curve. Reuters approaches but does not cross backoff.
6. **Convergence** (`neural_convergence.csv`, `neural_training_curves_100pct.png`).
   - 207 of 210 neural fits stopped on patience. ROCStories LSTM at 60/80/100% reached the 20-epoch cap; best epochs were 19/19/20. Do not call these converged.
   - Gutenberg FFNN selects epoch 1 and LSTM 1-3. WebText LSTM selects epoch 2 at 20% and epoch 1 thereafter. Validation-source sensitivity is plausible, but under-training is not established by these observations alone.
   - LSTM best epochs are 6-8 on Brown and Reuters, 9-11 on Inaugural. Longer training is a proposed ablation, not an inferred improvement.
7. **Cost.** At 100%, n-gram fitting is a single shared counting pass: 0.6 s (Inaugural), 8.6 s (Brown), 6.3 s (Gutenberg). LSTM training takes 22 s, 319 s and 188 s respectively.
   - On Brown, the LSTM costs about 37× the fit time for a 20% PPL gain.
   - Trigram-backoff *scoring* is slow on CPU (Gutenberg 153 s).
   - Added-corpus shared count/LSTM train seconds: Reuters 5.9/251.9; ROCStories 17.5/3,346.8; WebText 0.3/13.1. ROCStories' approximately 191-fold training-time ratio accompanies a 29.2% PPL gain, but is not a hardware-normalized efficiency estimate.
8. **Isolated rare n-gram lists** (`rare_set_ppl.md`). Each n-gram is scored with only its own (n−1)-word context.
   - The FFNN matches or beats backoff on Brown (trigram list: 587 vs 710).
   - Backoff wins on Gutenberg (2,320 vs 2,577) and Inaugural (828.5 vs 899.9).
   - Backoff wins on Reuters (1,788.8 vs 2,079.8) and WebText (9,122.4 vs 15,658.8); FFNN context 2 wins on ROCStories (1,076.5 vs 1,398.6).
   - The LSTM is handicapped because it loses its sentence prefix. Report this as a secondary probe only.

### Hypotheses (data-efficiency augmentation)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: backoff wins at the smallest budget | Mixed: wins on Gutenberg/Inaugural/Reuters; loses to the best neural model on Brown/ROCStories/WebText. | full budget curves |
| H2: relative neural performance improves with data | Endpoint improvement in five corpora, with WebText a counterexample and Gutenberg a final-step reversal. Do not extrapolate a universal threshold. | ratio and fixed-cohort tables |
| H3: sparse-context categories explain different model strengths | Supported descriptively: LSTM wins the unseen-bigram/retained-target category in all six. Seen-context and OOV rankings vary; ROCStories and WebText sharpen the exceptions. | category tables and additive NLL decomposition |

## 3. Mandatory caveats (Limitations section)

1. **Budget order.** Budgets are sorted-document prefixes, so data growth is confounded with genre, era and author shift: Inaugural is chronological, Brown is genre-ordered, Gutenberg is alphabetical. There is only one split and one budget order. Seeds vary only neural initialization and batch order.
2. **Changing support.** Vocabulary changes make cross-budget aggregate PPL a changing task. Fixed cohorts help but do not fix support, mapped histories or capacity.
3. **`min_freq` deviation.** Brown uses 2; all other corpora use 1. Five corpora lack positive `<unk>` targets, not all gradient updates. Do not treat the six-corpus comparison as a controlled threshold ablation.
4. **Tokenization.** Tagged punctuation, attached punctuation, news markup and web abbreviations affect vocabulary/OOV. Cross-corpus absolute PPL levels are not comparable measures of intrinsic difficulty.
5. **Document coverage and validation shift.** Gutenberg has two test books and two validation books: Whitman poetry and Chesterton's novel. WebText has one validation source and one test source. Early selection is observed, but its causal effect is untested.
6. **Fixed hyperparameters.** No neural tuning per corpus or budget, a fixed discount, and no Kneser-Ney. Parameter counts grow with vocabulary (about 0.84M-29.58M). Three ROCStories runs are epoch-capped.
7. **Hardware heterogeneity.** Runtime differences across corpora and budgets are confounded by hardware. For example, Gutenberg neural training is faster at 100% than at 80%.
8. **Rare-list probe.** It removes the sentence context, and the bigram and trigram lists contain different rows.
9. **Unequal replication.** Four corpora have three seeds; ROCStories and WebText have one. Do not manufacture standard deviations, claim seed robustness or mark all-corpus replication complete.
10. **Ethics and audit scope.** ROCStories near-duplicates were not scanned. Retained Reuters/Gutenberg near-duplicate flags and WebText personal-advertisement content require disclosure. Verify each source's license; public accessibility is not blanket redistribution permission.

## 4. PS-1 requirement checklist

| Requirement | Where | Status |
|---|---|---|
| Brown / WikiText-2 / synthetic | Brown full is present; five supplementary corpora | Brown covered; confirm supplementary-corpus eligibility against rubric |
| Unsmoothed bigram and trigram | `bigram_mle`, `trigram_mle` | done |
| Smoothed / backoff n-gram | add-one + absolute-discount backoff | done |
| Small neural LM | FFNN ctx1/ctx2, LSTM | done |
| PPL, vocab, OOV, runtime on the same test set | `main_results_100pct.md`, `budget_stats.csv` | done (state the hardware caveat) |
| ≥10 rare/unseen sequence analyses | 72 exported cases across six corpora; 15 individually interpreted in manuscript | drafted |
| Smoothing vs no-smoothing ablation | `smoothing_ablation.md`, `sparsity_vs_budget.png` | done |
| Which approach handles sparse data best, plus failures | six-corpus category/decomposition tables and counterexamples | drafted |
| ≥3 seeds, mean ± std | Brown/Gutenberg/Inaugural/Reuters: 13/42/2026 | partial: ROCStories/WebText have only seed 13 |
| Proposal page | — | to write (Section 7) |
| README: install, data, execution order, runtime, hardware, seed, outputs | `README.md`, `src/README.md` | add a V3 section (Section 5) |
| `data_access_instructions.md`, dataset card, contribution and AI-use statement | — | missing |

Fifteen cases interpreted in the working paper (all in `event_analysis_*.md`):

1. Brown *"vary according → to"*.
2. Inaugural *"important branch → of"*. Unseen trigram with a frequent bigram: add-one gives ln p −9.4 to −10.1 (≈1/V), backoff gives −0.05 to −0.10.
3. Brown *"comparable → basis"*: LSTM −1.67 vs backoff −9.40.
4. Gutenberg *"lead among → you."*: LSTM −5.25 vs backoff −10.30.
5. Inaugural *"partiality has → bestowed"*. Cases 3–5 show neural generalization to unseen bigrams.
6. Brown *"peanut and sesame → oils"*: LSTM ln p is −2.21 at 20% (where the target is `<unk>`, an easy class) but −20.03 at 100% (where it is a count-2 word). This illustrates the support change.
7. Gutenberg *"my lord → esau:"*.
8. Inaugural *"be duly → appreciated"*. Cases 7–8 are OOV targets without positive `<unk>` training examples; punctuation is attached in case 7.
9. Brown *"for god's → sake"*: −13.57 → −0.17.
10. Inaugural *"mr. vice → president,"*. Cases 9–10 are learned only once the relevant (modern) documents enter the budget.
11. Gutenberg *"o my → lord,"*: Bible test vs Shakespeare training.
12. Gutenberg *"frank churchill was one of the → boasts"*: LSTM, FFNN context 2 and trigram backoff are below -13; trigram add-one is -11.30, so not every model is below -13.
13. Reuters *"in 1985's 4th -> qtr."*: unseen bigram, target count 9; LSTM -5.90 vs trigram backoff -15.93.
14. ROCStories *"he had lost 5 -> pounds."*: unseen bigram, target count 106; LSTM -0.31 vs trigram backoff -12.01. Within-sentence generalization, not cross-sentence story reasoning.
15. WebText *"ala. lots -> of"*: absent trigram but bigram count 15; trigram backoff -0.05 vs LSTM -4.39, a non-OOV failure.

## 5. Submission archive layout

The layout below is a packaging proposal, not a claim that all deliverables exist. Use all six corpus names, the current `paper_v3_six_corpora_20261106` output folder and source-specific redistribution terms. The manuscript links both historical and current evidence, so include both or revise links for the submitted package.

```
<Group>_PS1/
├── research_paper.pdf
├── proposal.pdf                        # 1 page (Section 7)
├── README.md                           # V3: install, data prep, execution order, runtime, hardware, seeds, outputs
├── requirements.txt                    # from src/requirements.txt; pin torch/numpy/pandas/nltk versions actually used
├── data_access_instructions.md         # five NLTK corpora + archived ROCStories mirror, split manifests + SHA256
├── dataset_card.md                     # six corpora with source-specific rights and audit scope
├── CONTRIBUTIONS_AND_AI_USE.md         # role per member + what AI tools did/did not do
├── source_code/                        # src/: lm/, data-pipeline/, configs/, tests/, run.py, model_comparison.py,
│                                       #       prepare_mega_run.py, analyze_comparison_sweep.py
├── notebooks/                          # colab/comparison_sweep_frozen.ipynb (+ Kubeflow recovery script)
├── data/<corpus>/                      # split_documents.json, prepared_manifest.json, leakage_audit.json,
│                                       # near_duplicate_audit.json, rare-seq-*.csv (redistribute only if permitted)
├── results/
│   ├── <corpus>/metrics/               # metrics.csv, perplexity_by_percentage.csv, event_verification.json,
│   │                                   # artifact_manifest.json, charts.png
│   ├── <corpus>/train_<p>pct/          # config.yaml, data_stats.json, runs.csv, summary.csv,
│   │                                   # models/*/{*_history.csv,*_result.json}
│   └── paper_v3_six_corpora_20261106/   # current generated tables + audit
└── figures/                            # current six-corpus figures
```

- **Exclude and link instead** (Drive URL plus the SHA256 already in `artifact_manifest.json`):
  - `*.pt` weights and `checkpoints/`
  - `count_models.pkl.gz`
  - `predictions*.csv` (0.4–10 GB)
  - `scoring_events.jsonl.gz` (0.19–2.9 GB)
  - per-budget `train.txt` snapshots, which can be regenerated from the manifests
- **Do not present** the earlier pilots (`inaugural_convergence_v1`, `nltk_data_efficiency_pilot_v*`) as main results. Either drop them or put them in one Appendix paragraph labelled "preliminary pilot".
- **README must state:**
   - Seeds 13/42/2026 on four corpora; seed 13 only on ROCStories/WebText. Split seed 42 except WebText 3999; sentence-prefix budgets.
   - Brown trained on Colab; the remaining corpora include Kubeflow training/recovery. Preserve execution provenance rather than assuming uniform hardware.
   - Neural training totals: Brown 101.4, Gutenberg 127.2, Inaugural 9.2, Reuters 86.2, ROCStories 333.7 and WebText 2.6 minutes. The first four have 45 fits each, the last two 15 each; scoring is additional.
  - Execution order: prepare → train (`run.py`) → compare (`model_comparison.py`) → `analyze_comparison_sweep.py`.

## 6. Paper structure and positioning

**Current title:** *Smoothing, Sparsity and Vocabulary Growth: Comparing Backoff and Neural Language Models Across Training Budgets*. The working manuscript supersedes the earlier three-corpus outline. Confirm the required word limit before submission rather than forcing all six corpora into the original approximate 5,000-word target.

| Section | Words | Content pointers | Evidence |
|---|---|---|---|
| Abstract | 200 | Six corpora, nine models, five budgets, unequal seed coverage. Two aggregate LSTM wins, three OOV-driven reversals, WebText counterexample. | main table, loss decomposition |
| Keywords | — | n-gram smoothing, backoff, neural language model, LSTM, data sparsity, data efficiency, perplexity | |
| 1 Introduction | 550 | Descriptive six-corpus comparison; separate contextual novelty, lexical coverage and growing prediction support. Do not claim a controlled cross-corpus intervention. | |
| 2 Related Work | 500 | Foundational attribution plus 2020-2026 work: BabyLM baseline comparisons, Infini-gram, kNN-LM, ByT5, compute and exposure-order studies. | 17 references in manuscript |
| 3 Dataset and Preprocessing | 550 | Section 1 table; document-level splits, quarantine, near-duplicate audit; tokenization differences; per-budget vocab; budget order | `budget_stats.csv`, leakage audits |
| 4 Methodology | 650 | Equations: MLE, add-one, absolute-discount backoff with α normalization, FFNN, LSTM; target support (UNK/EOS in, PAD/BOS out); early stopping | `src/lm/statistical/ngram.py`, configs |
| 5 Experimental Setup and Metrics | 450 | Budgets, seeds, PPL definition (with EOS), accuracy, OOV, zero-prob rate, runtime; sparsity categories; fixed cohort; event verification | |
| 6 Results | 1,000 | **6.1** main table (PS-1); **6.2** smoothing ablation; **6.3** seeds and variance; **6.4** runtime/quality; **6.5** data efficiency: curves, ratio, fixed cohort (H1–H3) | `main_results_100pct.md`, `smoothing_ablation.md`, `ppl_vs_train_words.png`, `neural_vs_backoff_ratio.png`, `event_analysis_*.png` |
| 7 Error Analysis and Discussion | 900 | Fixed-comparator NLL decomposition, fifteen cases, WebText mapped-accuracy artifact, ROCStories seen-context advantage, conditional versus aggregate loss. | category tables, cases, accuracy audit |
| 8 Limitations, Ethics, Reproducibility | 450 | Single-seed extensions, small document samples, incomplete near-duplicate coverage, privacy/licensing, 390-condition audit and no new inference. | |
| 9 Conclusion | 200 | Conditional strengths, not a universal crossover or OOV-only explanation. Separate existing-result research from controlled reruns. | |
| References | — | ≥ 5 required; list below | |
| Appendix | — | Full per-budget tables, rare-list table, convergence table, all 36 cases, hyperparameters, environment | `results/paper_v3/tables/*` |

**Supported conclusion, not a mandated wording:** no family wins everywhere. LSTM beats both backoff orders on unseen mapped bigrams with retained targets in all six corpora. Aggregate rankings also depend on conditional OOV loss, its event share, seen-context performance and the selected evaluation population. WebText refutes universal improvement with budget; ROCStories refutes a universal backoff advantage on seen trigrams. No result proves how a tokenizer or stopping intervention would change performance.

### Reference handling

Use the manuscript's 17-entry bibliography, including ten entries from 2020 onward. The list below is the original planning inventory, not the current paper bibliography. It contains alternative editions and older candidates; do not merge it blindly or duplicate citations. Current publication metadata and abstract checks do not replace the outstanding full-text literature review.

1. S. M. Katz. Estimation of probabilities from sparse data for the language model component of a speech recognizer. *IEEE Trans. ASSP* 35(3):400–401, 1987.
2. H. Ney, U. Essen, R. Kneser. On structuring probabilistic dependences in stochastic language modelling. *Computer Speech & Language* 8(1):1–38, 1994.
3. R. Kneser, H. Ney. Improved backing-off for M-gram language modeling. *ICASSP* 1995, 181–184.
4. S. F. Chen, J. Goodman. An empirical study of smoothing techniques for language modeling. *Computer Speech & Language* 13(4):359–394, 1999.
5. J. T. Goodman. A bit of progress in language modeling. *Computer Speech & Language* 15(4):403–434, 2001.
6. Y. Bengio, R. Ducharme, P. Vincent, C. Jauvin. A neural probabilistic language model. *JMLR* 3:1137–1155, 2003.
7. S. Hochreiter, J. Schmidhuber. Long short-term memory. *Neural Computation* 9(8):1735–1780, 1997.
8. T. Mikolov, M. Karafiát, L. Burget, J. Černocký, S. Khudanpur. Recurrent neural network based language model. *Interspeech* 2010.
9. M. Banko, E. Brill. Scaling to very very large corpora for natural language disambiguation. *ACL* 2001.
10. J. Hestness et al. Deep learning scaling is predictable, empirically. arXiv:1712.00409, 2017.
11. J. Kaplan et al. Scaling laws for neural language models. arXiv:2001.08361, 2020.
12. W. N. Francis, H. Kučera. *Brown Corpus Manual*. Brown University, 1964/1979.
13. S. Bird, E. Klein, E. Loper. *Natural Language Processing with Python* (NLTK). O'Reilly, 2009.
14. D. Jurafsky, J. H. Martin. *Speech and Language Processing*, 3rd ed. draft, ch. 3 (N-gram LMs).
15. T. Gebru et al. Datasheets for datasets. *Communications of the ACM* 64(12):86–92, 2021 (for the dataset card).

## 7. Proposal page (one page)

- **Title:** as above.
- **RQ:** Which LM family best handles sparse data, and how does the answer change with the training budget?
- **Datasets:** Brown full, Gutenberg, Inaugural, Reuters, ROCStories, WebText; the last two are single-seed exploratory extensions.
- **Baselines:** unsmoothed bigram and trigram.
- **Advanced methods:** add-one and backoff; FFNN; LSTM.
- **Metrics:** PPL, OOV, vocabulary size, top-1 accuracy, zero-probability rate, runtime.
- **Ablations:** smoothing vs none; data budget 20–100%.
- **Risks:** OOV handling, domain shift, compute quota (actually happened: Colab quota → Kubeflow recovery), changing vocabulary.
- **Role allocation:** fill in per member.

## 8. Research Topics Using the Existing Results

**Recommendation:** prioritize Topic 1 as a measurement-focused extension and Topic 2 as an algorithmic extension. Topic 3 is a small, well-supported companion analysis. Do not split the same descriptive results into multiple near-identical papers: a separate study needs a distinct question, added analysis and literature positioning. No new experiments below were executed in this update.

### 8.1 Vocabulary Growth and Model-Ranking Reversals

**Question:** When does vocabulary expansion change what a language-model learning curve measures, and which event groups reverse model rankings?

**Evidence already present:** Reuters/Gutenberg/Inaugural lose in aggregate but win on non-OOV events; ROCStories survives extreme conditional OOV loss because OOV prevalence is low; WebText loses on both groups. Brown's aggregate and fixed-cohort trends disagree.

**Next analysis, no base-model training:** align saved events by corpus, split, sentence index and position. Track always-retained, newly-retained and still-OOV targets across budgets; separate EOS. Report event counts, within-budget NLL differences, vocabulary growth and changes in model ranking. Compare loss on fixed cohorts with full-test loss, keeping changing target identity explicit. A hypothetical reweighting of known/OOV groups can show a ranking-sensitivity threshold, but is not the result of changing tokenization or retraining.

**Artifacts:** scoring-event archives, budget vocabularies, training snapshots, category/cohort tables and split occurrence manifests. **Effort:** low to medium; offline analysis only. **Potential contribution:** an evaluation protocol and empirical taxonomy of ranking reversals across six contrasting corpora. The current decomposition is the starting point, not a new result to claim twice.

### 8.2 Validation-Selected Backoff/Neural Mixtures

**Question:** Can a probability mixture retain count-based reliability while using neural gains on novel contexts?

For a common budget and prediction support, evaluate $p_{mix}(w|h)=\lambda p_{neural}(w|h)+(1-\lambda)p_{backoff}(w|h)$. Select the backoff order and mixture weight using validation NLL only, freeze them, then report test NLL/PPL by category and seed. Compute mixtures with stable log-sum-exp rather than averaging perplexities or log probabilities. Compare against both components and a validation-selected single-model baseline.

**Feasibility:** saved validation/test true-target probabilities are sufficient for mixture NLL/PPL, without loading weights or retraining base models. Align all event identities and vocabularies; deterministic backoff pairs with each neural seed. Mixture top-1 accuracy cannot be recovered from true-target probabilities and component top-1 labels alone.

**Extension:** a context-only gate using training-history frequency or context coverage. Do not deploy a gate using whether the true target is OOV or its hindsight sparsity category: that information is unavailable before prediction. Such a target-aware selector is only an oracle diagnostic. Fitting a gate is a new training task requiring an agreed validation protocol, not part of this update.

**Effort:** medium. **Potential contribution:** selective combination with interpretable failure groups. Reuters is a strong three-seed starting point; ROCStories tests whether a mixture can help an already strong neural model, while WebText tests whether it should mostly defer to counts. Current test results have already informed the proposal, so disclose exploratory selection; independent confirmation requires untouched evaluation data.

### 8.3 Unknown-Label Accuracy Versus Lexical Prediction

**Question:** When do accuracy and PPL imply opposite conclusions because many words collapse into `<unk>`?

**Evidence already measured:** WebText trigram add-one's 41.41% mapped accuracy contains only 28 retained-word successes; 1,442 successes are unknown labels. Its retained-word accuracy is 1.55%, below trigram backoff's 5.10%.

**Next analysis:** extend the saved-event correctness audit to every corpus/budget/model. Report mapped accuracy, retained-word accuracy, OOV-label accuracy and EOS separately, with exact denominators. Add within-budget accuracy/PPL ranking comparisons and an explicit unknown-label baseline. Never interpret correct `<unk>` as spelling recovery.

**Artifacts:** `correct`, `oov`, `eos`, `prediction_id` and target fields in saved scoring events. **Effort:** low. **Potential contribution:** an evaluation-audit note or companion section. Do not call this a full calibration study: target log probability is not maximum predicted confidence, and ordinary ECE or multiclass Brier scores need additional distribution information not present in these event records.

### 8.4 Rare Words, Novel Contexts and Concentrated Loss

**Question:** Are neural gains attributable to rare words, new combinations of common words, or a small tail of extreme prediction errors?

**Next analysis:** join raw training unigram/bigram/trigram counts to saved target events. Separate zero count, singleton, low-frequency and common targets; distinguish retained from OOV and raw from mapped novelty. Report category mass, paired NLL differences and the share of total loss contributed by the worst 1%, 5% and 10% of events. A few selected examples do not establish prevalence.

**Artifacts:** exact budget texts, saved count tables, event log probabilities and occurrence-to-document mappings. Reading existing counts or tallying descriptive frequencies does not require a new LM fit. **Effort:** medium. **Potential contribution:** a sharper account of what "generalization to sparse language" means. For uncertainty, resample paired document groups where sufficiently many exist, keeping seed variation separate. WebText's one test document and Gutenberg's two books cannot support broad document-population confidence claims.

### 8.5 Within-Sentence Context and Architecture Sensitivity

**Question:** Do LSTM gains vary with sentence length, target position and local-context support, relative to one- and two-token FFNNs?

**Next analysis:** stratify existing paired losses by sentence length/position, known-history status and target frequency; compare FFNN context 1, FFNN context 2 and LSTM at each fixed budget. Analyze full-sentence and isolated n-gram probes separately because they score different target populations and available contexts.

**Artifacts:** event positions, EOS boundaries, test text and raw training counts. **Effort:** medium. **Potential contribution:** descriptive localization of context benefits and failures, particularly ROCStories versus WebText. This cannot isolate a causal long-context benefit because model families, capacity and optimization differ. A matched-weight prefix-truncation ablation requires new checkpoint inference; cross-sentence coherence or story-completion claims require a different experiment.

### 8.6 Stopping Policy and Measured Quality/Cost Frontiers

**Question:** Which model-budget choices are Pareto-efficient in observed validation-selected quality and measured runtime, and which are limited by the epoch ceiling?

**Next analysis:** combine final metrics, parameter counts, fit/scoring times and histories. Mark dominated choices, stopping reason and selected epoch; report held-out quality only after a validation-based selection rule. Treat the 207 patience stops and three ROCStories caps separately. Use final saved train/validation metrics for checkpoint gaps; training-epoch averages were collected while weights changed and are not equivalent to final-checkpoint rescoring.

**Effort:** low for descriptive frontiers. **Limit:** timing varies with hardware, epoch count and seed coverage. Do not infer FLOPs, energy, carbon cost, a hardware-independent scaling law or guaranteed gains beyond epoch 20 from elapsed seconds. A stronger efficiency study needs standardized reruns; longer ROCStories training is a new controlled experiment.

### 8.7 What Requires New Experiments

| Proposed claim | Missing control or evidence |
|---|---|
| ROCStories/WebText rankings are robust to initialization | At least two additional neural seeds per model/budget, keeping the existing split/config fixed |
| More data itself causes the improvement | Fixed prediction support and capacity, alternative randomized document/budget orders, multiple splits |
| Training `<unk>` fixes the failures | Uniform threshold or controlled unknown-word exposure; matched retraining and explicit target/support policy |
| Subwords or bytes solve this OOV problem | New tokenization and training; compare using a common evaluation unit, not raw word-vs-byte PPL |
| Longer context causes LSTM's gains | Matched checkpoint context truncation or matched-capacity/context experiments |
| Models transfer across domains | Train-on-A/test-on-B evaluation with compatible support and multiple held-out sources; six separately trained corpora are not cross-domain transfer experiments |
| The neural model has saturated | Controlled epoch-budget extension for capped runs; validation selection and untouched evaluation protocol |

**Suggested order:** finish Topic 1's cross-budget transition analysis, then run the validation-only mixture selection in Topic 2, with Topic 3 as a concise metric audit. Keep Topics 4-6 as optional extensions to a coherent paper rather than expanding the experiment matrix immediately. No new model training, rescoring or downloads are needed for Topics 1, 3 and the descriptive portions of 4-6; Topic 2's constant-weight mixture can reuse saved scores but introduces a new selection experiment.

## Manuscript Rewrite Completed (7 November 2026)

All prose in [paper_working_draft.md](paper_working_draft.md) has been rewritten in the approved abstract's concise, direct research style. The abstract remains verbatim at 124 words; all seventeen reference entries remain byte-identical and cited. The manuscript has nine numbered main sections, separate Methodology and Experimental Setup sections, fifteen individual case interpretations and three appendices. The version-0.4 warning, pending-contribution language and Working Notes are removed. Historical analysis and corrections remain in this existing plan; obsolete pending-task lists were not restored.

Final main length is **4,728 text words from the Pandoc AST**, including title, roster, headings, tables and captions but excluding mathematical notation, image descriptions, references and appendices. A conservative Pandoc plain-text count including equation representations and table layout is **4,798 words**. Both are within the requested 4,300-4,800 target and the 4,000-5,500 requirement. The main count was checked immediately after the first substantive edit, then repeated after each length adjustment. Abstract SHA256 remains `7b1a4821431cdce7effed3942bb91bf308efde824e276bb151b4890a39e080c2`; reference-block SHA256 remains `09252f7c2febf3dd0a69427b032d262a9061afac4bac461069fcfff5b2559aef`.

Figures 1-3 retain the actual pipeline, phase-to-code and architecture assets. Figures 4-7 use the completed PPL, vocabulary/OOV, accuracy/UNK and loss-decomposition PNGs. Appendix A retains all three actual Torch screenshots as A1-A3. Appendix B uses recorded-time and event-share PNGs as B1-B2, with runtime and conditional-category tables. Appendix C transcribes all thirty actual budget sizes. All six final PNGs and their PDF counterparts are present; no assets were regenerated.

### Verification and Numerical Corrections

Pandoc parsed the manuscript successfully. A read-only Ruby CSV/Pandoc check passed **385 assertions** covering 54 full-budget PPL cells and available sample SDs, all thirty budget rows, split sentence/word sizes, vocabulary/OOV, thirty neural/backoff ratios, additive loss contributions, all seventeen citations, thirty local links and twelve figure/image pairs. Manuscript diagnostics reported no errors. Verification used named small CSV/JSON/docs and file-existence/size checks, not scoring-event or count archives.

The checked authorities are [full-budget and budget summaries](../../results/paper_submission_20261107/tables/test_metrics_seed_summary.csv), [actual budget sizes](../../results/paper_submission_20261107/tables/budget_data_sizes.csv), [seed-mean loss contributions](../../results/paper_submission_20261107/tables/full_budget_loss_seed_summary.csv) and [Laplace attribution](../../results/paper_submission_20261107/tables/laplace_tie_attribution.csv). Separate receipt checks confirm 390 condition rows, 270 summaries, thirty budgets and 210 convergence rows. All **524 existing graph flags** in [graph_checks.json](../../results/paper_submission_20261107/graph_checks.json) are true; its six PNGs total 903,209 bytes. Its saved nonblank/text-bound checks are reused, not represented as a fresh render or pixel inspection. The [accuracy receipt](../../results/paper_accuracy_20261107/accuracy_checks.json) confirms 390 audited conditions with zero fitting or checkpoint loading in that prior audit.

Gutenberg's full-budget LSTM deficit is correctly rounded to **15.3%**, not 15.2%. WebText trigram add-one retains 1,470/3,550 overall successes, 28/1,803 retained-word successes and 1,442/1,747 OOV-label successes. Its actual/always-UNK accuracies are 41.4085%/49.2113%; unseen uniform-tie contribution is 40.6197 percentage points; abstention/random-tie expectations are 0.7887%/0.7972%. These are accuracy-only diagnostics, not changed probabilities or PPL. Brown's 1,349 seen-history UNK successes remain explicitly separated into 1,139 unique maxima and 210 tied maxima.

The paper now records all three authors' equal contributions and the group-reported instructor approval confirmed on 7 November 2026, without inventing affiliations, an instructor name, a written approval or an original approval date. The completed [dataset card](../submission/dataset_card.md), [data access](../submission/data_access_instructions.md), [contribution/AI disclosure](../submission/contributions_and_ai.md) and [reproducibility document](../submission/reproducibility.md) are linked. Retained Gutenberg/Reuters near-pairs, missing ROCStories near-screening, single-seed limits, heterogeneous hardware and ten absent count archives remain disclosed. The 210 usable final checkpoints do not imply complete replay; the zero-byte resume file is excluded.

Only the manuscript and this appended plan record were edited. No source, README, supporting document or result file was changed. No model fitting, training, checkpoint access, inference, rescoring, archive scan, install, download, commit or revert occurred. Google Docs, exports and archive packaging were not touched; ZIPs remain deferred. These current completion facts supersede earlier historical statements that the abstract, contributions, supporting documents or broad rewrite were still pending, without rewriting that history or claiming final export or all-author sign-off.

## Shared Document and Final Exports (7 November 2026)

The original shared Google Doc now contains all nine rewritten main sections and three appendices. Before replacement, its complete text matched the last verified Step 7 snapshot, protecting intervening collaborator changes. A named pre-rewrite version was saved. The replacement was made in Editing mode, without comments or suggestions, and its complete text matched the verified DOCX import. Saved text is 43,573 characters, SHA256 `12781c134f9f5fa0d00eabb9c033ba4caf377ff347432143821827bbf008fc47`. The approved abstract remains verbatim at 124 words; Working Notes are removed.

The final [PDF](research_paper.pdf) has 26 readable pages, no blank pages and the expected abstract, conclusion and appendices. The [DOCX](paper_working_draft.docx) has twelve embedded figures, thirty-four native equations and eleven tables; archive CRC checks pass. The [offline print source](research_paper.html) and [print stylesheet](paper.css) preserve literal token symbols and render MathML. Representative corpus-table, backoff-equation and PPL-chart pages were visually inspected. Table text fits corpus names, duplicate generated image captions are removed, and chart legends are separated from axis labels.

The final figure generator was rerun against named small CSV/JSON inputs. All 524 checks pass for 390 conditions, 270 seed summaries, thirty budgets and 210 convergence records. Six final PNGs total 896,378 bytes. A new legend-axis clearance check passes; relative output paths work. Eleven fit-free `AccuracyAnalysisTests` pass, and diagnostics/whitespace checks are clean. No fitting, training, checkpoint loading, inference or rescoring was performed.

All supporting documents and invocation guides are complete. The user explicitly deferred ZIP packaging. A later upload to refresh only the shared Doc's table/image layout was blocked by Drive HTTP 429; the prose rewrite is already saved and the local final exports include the layout fixes. Fresh-machine replay, independent upstream rights certification and final all-author sign-off remain unverified. Neither instructor approval nor consistency checks remove the disclosed scientific limitations.
