# Recommendations: Validation-Based Stopping Follow-Up

Date: 2026-10-29

Status: Proposed follow-up, not an executed experiment or authorization to change data. These recommendations build on the completed three-corpus pilot and the read-only review of 2026-10-27. The pilot configuration was checked again on 2026-10-29; it still specifies three epochs and one final neural seed. Earlier implementation findings must be rechecked against current code before acting on them.

## Recommendation

Use **Inaugural** for the next bounded experiment with validation-based stopping, subject to corpus eligibility and data-audit checks. Preserve the completed pilot and run any approved follow-up under a new configuration, output directory, and experiment identity.

The next step should establish whether the observed neural results persist with more training and multiple initializations. It should not expand immediately to seven corpora, a large hyperparameter search, or additional architectures.

**Assignment caveat:** PS-1 explicitly names Brown, WikiText-2, or documented synthetic data, although the assignment also contains a broader allowance for public data. Obtain instructor confirmation before using Inaugural as the main submission corpus. If that confirmation is unavailable, use Brown for the primary study and retain the current results as supplementary pilot evidence. Coordinate the selected corpus variant with Groups 6 and 11.

## Why Inaugural

| Criterion | Evidence from the completed pilot | Implication |
| --- | --- | --- |
| Independent document groups | 43 training, five validation, and 12 test documents | More useful document-level coverage than Gutenberg's 13/1/4 or WebText's 2/1/3, although still a small sample. |
| Training progression | Full-budget LSTM validation perplexity was approximately 253, 203, and 180 over epochs one to three | The model was still improving when the three-epoch ceiling stopped training. |
| Computational scale | Approximately 106,000 training words at full budget | Longer training and repeated seeds are practical candidates for a bounded follow-up; actual runtime must be measured. |
| Architecture comparison | Full-budget test perplexity was 177.10 for LSTM and 174.64 for FFNN | The small, single-seed difference needs replication before interpreting the ranking. |

Evidence: [Inaugural partition manifest](../../../results/colab_imports/20261026T175208Z/nltk_data_efficiency_pilot_v2/prepared/inaugural/manifest.json), [LSTM validation history](../../../results/colab_imports/20261026T175208Z/nltk_data_efficiency_pilot_v2/inaugural/budget_100/final/lstm/seed42/history.csv), and [pilot summaries](../../../results/colab_imports/20261026T175208Z/nltk_data_efficiency_pilot_v2/summary.csv). These private local artifacts are not included in a fresh Git clone.

Gutenberg is a useful later comparison, but its single validation document makes stopping decisions sensitive to one text. Its document budgets also contain large token-count jumps. WebText should remain a diagnostic case until its duplicate weighting and unknown-token exposure are addressed: ten nominal budgets represent only two training subsets, and each subset has just two training tokens mapped to `<unk>` despite roughly 42-51% test OOV.

## Research Question

> Under a growing, training-only vocabulary, do the observed LSTM and FFNN performance patterns persist when training uses validation-selected stopping and multiple random initializations?

Secondary questions are whether the three-epoch ceiling materially restricted learning, whether stopping behaviour changes with training budget, and whether model differences are concentrated in known-word, OOV, or sparse-context events.

The experiment need not produce a neural winner, a statistically clear architecture ranking, or lower perplexity at every larger budget. A finding that the rankings are unstable, or that longer training does not improve held-out prediction, is informative when the controls are documented.

## Proposed Experiment Matrix

| Setting | Recommendation |
| --- | --- |
| Corpus | Inaugural, conditional on approval and audit gates |
| Training budgets | 10%, 50%, and 100% of the eligible training-document pool |
| Neural models | Existing LSTM and context-2 FFNN |
| Final training seeds | 42, 43, and 44 for each model and budget |
| Maximum epochs | 20 |
| Early-stopping patience | Three consecutive validation checks without improvement |
| Selection metric | Corpus token-mean validation negative log-likelihood; lower is better |
| Final weights | Checkpoint with the best validation metric |
| Data and budget seeds | Retain 42 and 55, respectively, unless an approved preparation amendment requires a different protocol |
| Vocabulary | Retain `per_budget`, training-only construction, `min_freq=2`, and one shared vocabulary per budget |
| Other settings | Keep the pilot architectures, learning rate, batching policy, optimizer settings, dropout, clipping, and scoring support unchanged unless a separately justified change is declared |

This is **18 neural fits**: three budgets, two architectures, and three training seeds. The seed values are proposed specifically for this bounded follow-up; the broader [plan.md](plan.md) describes a different, larger matrix. This recommendation does not silently amend that plan or the existing [pilot configuration](../../../src/configs/data_efficiency_nltk_pilot.yaml).

Retain unsmoothed bigram, unsmoothed trigram, and validation-selected add-k trigram results for the same three budgets so that all required PS-1 systems remain represented. With the current three-candidate add-k search, fresh statistical runs add nine final fits and nine validation-only candidate fits. Reuse archived measurements only if data, splits, vocabulary, code, and scoring identities are demonstrably unchanged. A duplicate-repair amendment normally requires new statistical fits as well.

## Stopping and Checkpoint Rules

1. Start every model, budget, and seed from a fresh initialization. Do not warm-start the 50% model from the 10% model or continue a pilot checkpoint as the primary fresh-run comparison.
2. Evaluate the complete validation partition after every epoch in evaluation mode, without dropout or gradient updates.
3. Sum validation negative log-likelihood over scored targets and divide by their total count. Do not average unweighted batch or sentence losses.
4. Save the best checkpoint and reset the patience counter when the declared improvement rule is met. Increment the counter otherwise, stopping after three consecutive non-improving checks.
5. Use strict improvement as the initial proposed rule. If a positive minimum-improvement threshold is chosen instead, document its units and value before final testing; do not imply that a configuration field exists without verifying trainer support.
6. Stop at 20 epochs even if patience has not expired. Label a run that is still improving at the ceiling as budget-limited, not converged. Do not extend only the runs that have unfavourable test scores.
7. Restore the best validation checkpoint before final scoring, and record both its epoch and the last epoch executed.

Validation NLL and perplexity select the same checkpoint when computed from the same targets; NLL avoids exponentiation overflow. Three epochs with patience three offer little opportunity to distinguish early stopping from simply reaching the epoch ceiling.

Different seeds can stop at different epochs. That is consistent with a shared stopping policy, but means training compute differs. Record the actual cost rather than describing the comparison as equal-compute.

## Checks Before a New Run

### Corpus and Split Integrity

The 2026-10-27 archived-data audit found four normalized Inaugural sentence types shared between full-budget training and test. All were shorter than eight words. This does not establish extensive passage leakage, but it does not satisfy the assignment's strict no-repeated-sentence requirement. Document-level source separation alone is insufficient.

Define and approve a deterministic repair policy covering training, validation, and test before launching. Audit exact sentence overlap at the model's lowercase/whitespace normalization, plus near-duplicate sentences and documents using a documented rule. Do not assume the previous document-level MinHash screen establishes sentence-level cleanliness.

Preserve original imports and the completed pilot. Prefer a documented training-pool quarantine/removal policy that preserves test bytes; if a new split or changed test population is necessary, require an explicit protocol amendment. Keep document provenance, record excluded items and revised counts, and confirm that enough validation and test documents remain. Never silently redraw splits to improve model results.

Changing preparation means the follow-up is not an epochs-only replication of the pilot. State that distinction and avoid attributing every performance difference to stopping behaviour.

### Implementation and Reproducibility

Re-run the existing tests against the current checkout. The 2026-10-27 review recorded 13 passes and four failures: three missing CLI table-renderer errors and a sentence-boundary inconsistency. That is a historical finding, not a fresh claim about the latest pulled code. The archived suite bypassed the failed renderer.

Before relying on interactive comparisons, verify that suite checkpoints recover the correct document budget rather than defaulting to all training text. Confirm consistent sentence boundaries, BOS/EOS handling, padding exclusion, and prediction support across the suite and any analysis tools.

Bind prepared sentence content and concrete partition membership to run identity, not only preparation settings. Verify best-checkpoint restoration, early-stopping counter behaviour, seed application, and resume identity with focused tests. Optional backoff/Kneser-Ney implementations require independent correctness checks before use; they are not required for this follow-up.

Create a new configuration and output root only after these decisions are approved. Preserve environment versions, data manifests, vocabulary identities, code identity, and checkpoints. For Colab, generate a matching new bundle and checksum rather than resuming the completed pilot with altered settings.

## Evaluation and Interpretation

For every model and budget, report full-test perplexity, vocabulary size, OOV rate, fitting time, and scoring time. For neural models, retain each seed's result and report mean and sample standard deviation over the three seeds. Also report selected epoch, executed epochs, parameter count, actual training tokens/events processed, and any resource ceiling reached.

Separate known-word, OOV, and sparse-sequence losses, with event counts and definitions. Distinguish unseen combinations of retained words from combinations containing `<unk>`. Report known-word accuracy separately if accuracy is shown; a correct `<unk>` prediction does not recover the original lexical word. Timing must distinguish fitting, epoch validation, tuning, and final scoring where measured, and identify CPU/GPU differences.

Compare LSTM and FFNN within the same corpus/budget using their shared vocabulary and target stream. Across budgets, vocabulary expansion changes target mappings, output support, and parameter counts. A larger-budget perplexity increase does not by itself show that more data hurts learning. Conversely, falling OOV does not by itself prove improved prediction on familiar words.

Retain infinite MLE perplexities when zero-probability events occur, together with the zero-event rate. Keep the matched unsmoothed-versus-smoothed trigram comparison as the explicit smoothing ablation. Claims about neural superiority must be restricted to the statistical methods actually tested.

The pilot test results have already been inspected. This follow-up is therefore informed by exploratory evidence, not a pristine preregistered confirmatory study. Freeze the follow-up settings before further testing, use validation only for stopping and selection, and disclose the history. Stronger independent confirmation would require a separately agreed untouched evaluation population, not repeated tuning on the existing test scores.

Three training seeds estimate initialization/training variability on the fixed data split. They do not measure uncertainty from document selection, validation composition, or corpus choice. Do not present repeated seeds as independent datasets or small mean differences as established superiority.

## Required Research Outputs

- A versioned protocol and data audit stating whether preparation changed from the pilot.
- Train/validation learning curves for every neural seed, identifying best and final epochs. Training histories are accumulated while weights change; do not treat them as fixed-checkpoint training-set rescoring.
- Per-budget model comparisons with all seed results, aggregate variability, vocabulary/OOV statistics, and measured cost.
- Known-word, OOV, and sparse-context analyses with explicit denominators and original-versus-mapped token definitions.
- At least ten held-out rare/unseen sequence analyses selected by a documented category-based rule, including failures and cases where the preferred aggregate model loses.
- An explicit smoothing ablation and a conclusion limited to the tested corpus, vocabulary policy, architectures, and training procedure.

For each qualitative case, show original context, target word, mapped target, relevant training counts, model probabilities, and an explanation of what the case does and does not demonstrate. The saved first-ten event examples are not a substitute for this analysis.

Document dataset source, exact version or checksum, access date, distribution terms, preprocessing, split procedure, sizes, and limitations. Retain the assignment's scholarly-reference, borrowed-code, pretrained-resource, and AI-use declarations. Corpus availability through NLTK does not imply a common licence.

## Decision After the Follow-Up

If most runs reach a validation plateau and model rankings are reasonably stable across seeds, proceed to deeper error analysis and a carefully qualified comparison. If many runs still improve at epoch 20, report the ceiling and plan any further training change from validation evidence under a new declared protocol.

If LSTM and FFNN differences vary substantially across seeds, report that neither architecture has a reliable advantage under the tested conditions. If OOV effects dominate, prioritize understanding lexical coverage rather than enlarging the architecture sweep.

Use Gutenberg as a later robustness check only when its validation coverage and document-composition limitations are addressed. Defer WebText as a primary data-efficiency corpus. Additional corpora or a fixed-vocabulary diagnostic are separate scope decisions, not prerequisites for completing this bounded experiment.

**Bottom line:** continue with an approved, audited Inaugural follow-up using validation-selected checkpoints and three neural seeds. The goal is a better-supported explanation of the pilot, not a predetermined winning model or a larger run count.