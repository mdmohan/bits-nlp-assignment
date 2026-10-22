# Direction 3: Generalization to Unseen Combinations of Familiar Words

## Research Focus

**Working title:** Beyond Familiar Phrases: Evaluating Language Models on Unseen Combinations of Known Words.

This direction investigates whether a language model can predict a word in a combination it has never observed, even when the individual words are familiar. It separates vocabulary novelty from sequence novelty and asks whether access to additional preceding context helps resolve the latter.

The proposed intervention is controlled neural context length: compare the same small recurrent architecture using only a short history with versions that can inspect a longer prefix. Statistical models remain essential baselines.

The study measures a restricted form of generalization. Success on unseen n-grams alone is not proof of human-like compositional reasoning or systematic understanding. This is a proposed experiment, not a report of established results.

## Problem and Motivation

Suppose training text contains the words `committee`, `rejected`, and `proposal`, but never the sequence `committee rejected proposal`. A failure on this sequence could arise from unfamiliar word combinations rather than unknown vocabulary.

An unsmoothed trigram cannot assign positive probability to an unobserved target trigram. A smoothed model can, but nonzero probability does not guarantee an informative prediction. A neural model might use shared representations or a longer preceding context, but it might also rely heavily on familiar local phrases.

Ordinary aggregate perplexity mixes frequent phrases, rare combinations, and OOV events. It can therefore conceal the exact behaviour this project wants to study.

The research opportunity is to define the evaluation cases precisely and test a particular mechanism: whether information beyond the last two words helps predict unseen combinations of otherwise familiar words.

## Research Questions and Hypotheses

**Primary question:** Does extending a small neural language model's available history improve prediction of unseen combinations of known words beyond what a smoothed trigram and a short-context neural model achieve?

**Secondary questions:**
- How different are errors on unseen combinations from errors involving OOV words?
- Does longer context help all unseen combinations, or only those with useful earlier cues?
- Are apparent gains explained by target-word frequency rather than sequence generalization?
- Does longer context improve prediction enough to justify its additional computation?

**Testable hypotheses:**
- H1: Unseen combinations of known words remain harder than frequency-matched seen combinations.
- H2: Longer neural history reduces loss on a predefined unseen-combination subset.
- H3: The benefit is more visible when earlier words provide relevant information than when the preceding context is uninformative.

H3 is exploratory unless cue availability can be defined independently of model outcomes. Do not label examples as having useful cues merely because the long-context model succeeded.

## Recommended Data Design

Use a natural-text corpus such as a documented Brown genre subset. Preserve document and sentence boundaries. Agree on a distinct corpus variant with Groups 6 and 11.

Split documents into training, validation, and test sets before fitting models. Audit exact and near-duplicate sentences/documents and resolve overlap before freezing partitions. Preserve the raw corpus and record all preprocessing and duplicate decisions.

Build the vocabulary from training data only. Map rare training words and subsequent out-of-vocabulary words to `<UNK>` using one documented policy. Preserve original tokens and counts for defining the evaluation strata.

The main test set remains the full held-out natural-text set. Define additional diagnostic strata with a rule fixed before final evaluation. Do not repeatedly redraw the split until it makes a particular model look better.

### Operational Definition of Novelty

For an evaluated target and its two preceding real words, consider the resulting raw trigram:

- **Frequent seen combination:** its training count exceeds a predefined threshold.
- **Rare seen combination:** its training count is positive but small, for example one or two.
- **Unseen known-word combination:** its training count is zero, but all three words remain in the model vocabulary.
- **OOV-containing combination:** at least one of the three raw words maps to `<UNK>`.

The exact thresholds are design choices, not assignment requirements. Define them before final testing. Exclude sentence-boundary positions from this three-real-word diagnostic while still scoring those positions in full-test perplexity.

A stricter familiar-word subset can require each constituent word to have at least a chosen training frequency. This distinguishes an unfamiliar combination from a combination made entirely of individually rare words. Counts must come from training data, never from validation/test frequency estimates used to change the model.

Report stratum sizes. If the strict subset is too small, that limits the study; do not weaken criteria after seeing model results without treating the change as exploratory and using fresh evaluation data.

### Controlling Other Sources of Difficulty

Compare unseen and seen events within target-frequency bands. Also inspect history frequency, sentence position, and genre. Matching or stratification can reduce these confounds but does not eliminate all linguistic differences.

For a sentence-level analysis, retain the original context rather than evaluating isolated trigrams only. The long-context intervention requires genuine preceding information from the same sentence.

## Synthetic Alternative

A documented synthetic corpus can provide clearer control over which lexical or grammatical factors are recombined. It requires a generator, grammar or templates, vocabulary, seed, label/target rules, difficulty controls, optional noise process, examples, and independently constructed final test items.

The assignment explicitly forbids template leakage across training and final test. Different random seeds or different noun substitutions in the same template are not sufficient. Hold out complete template families and audit both exact and near-duplicate outputs. Explain which grammar rules or linguistic primitives are deliberately shared and which generation templates are disjoint.

Holding out templates also introduces structural novelty, so it is not a pure lexical-recombination test. Report those factors separately rather than claiming an isolated effect. Natural-text strata are the lower-risk primary option; synthetic data is appropriate only if generator design is itself a deliberate research contribution.

## Systems and Proposed Intervention

| System | Research role |
|---|---|
| Unsmoothed bigram | Required baseline and local-count failure reference. |
| Unsmoothed trigram | Required baseline directly exposed to unseen trigram events. |
| Smoothed trigram | Required nonzero-probability comparison. |
| Short-context LSTM | Small neural comparison restricted to the previous two words. |
| Longer-context LSTM | Proposed intervention, for example histories up to 8 or 32 words. |

Use the same embedding size, recurrent hidden size, output vocabulary, and architecture across neural context conditions. This keeps parameter count constant while changing available information. The short-context LSTM is still a learned neural predictor rather than an n-gram count model.

For a clean implementation, construct one next-word example per eligible target with a preceding history capped at the condition's limit. Reset recurrent state for each example and never carry state across windows or documents. Otherwise a nominal two-word condition might accidentally retain older information.

Every condition must train and evaluate on the same target positions, including short sentence prefixes with whatever history is actually available. Do not discard more targets for the longer-context condition. Use padding masks so padding does not become linguistic context or a scored target.

Training separate context-length conditions is the primary controlled comparison. Simply truncating history at test time for a model trained on long histories measures test-time context removal, which is useful but answers a different question.

## Experimental Protocol

1. Freeze natural-text splits, vocabulary, novelty definitions, frequency bands, and target-position alignment.
2. Verify that the known-word diagnostic subset contains no `<UNK>` inputs or targets in its defining trigram.
3. Train the statistical baselines and short/long-context neural conditions.
4. Select smoothing strength and neural training settings using validation data only, with a consistent tuning allowance across conditions.
5. Use at least three neural seeds where feasible. Keep the dataset fixed and match seeds across context conditions for paired reporting.
6. Freeze configurations and score all required models on the full test set and predefined diagnostic strata.
7. Analyse failures, including cases where longer history is distracting or makes no measurable difference.

A minimal experiment uses two context limits, such as 2 and 32. An intermediate limit is optional. Sliding-window training can cost more than a standard streaming LSTM implementation, so benchmark a small pilot and report the actual procedure rather than assuming equal compute.

## Metrics and Required Ablation

Report full-test perplexity, vocabulary size, OOV rate, and training/evaluation runtime for every system. Add per-stratum token negative log-likelihood or perplexity, event counts, and zero-probability rates.

Compute corpus and stratum perplexities from total negative log probability over their respective target counts. Do not compare perplexities across different vocabulary mappings as if they describe the same task. Keep sentence-ending targets and padding policies identical in full-test scoring.

Unsmoothed models can have infinite perplexity, particularly on unseen-combination strata. This is an expected property of the estimator, not evidence that the evaluation should skip those examples. Report the zero-probability rate alongside infinity.

**Primary ablation:** reduce the neural history from the longer window to two words while preserving architecture, vocabulary, target examples, and training protocol. This tests access to earlier context.

**Supporting smoothing comparison:** unsmoothed versus smoothed trigram on the same data. This distinguishes assigning nonzero probability from exploiting a longer linguistic history.

## Qualitative Error Analysis

Analyse at least ten actual held-out rare/unseen sequences. For the main research claim, include multiple unseen combinations whose constituent words are all known. Include OOV cases separately to explain why they are a different failure mode.

For each case, show the full relevant prefix, target, unigram and trigram training counts, novelty category, model probabilities, and whether longer history helped or hurt. Discuss plausible cues such as a previously mentioned topic, grammatical agreement, or a repeated entity, without claiming that one successful prediction proves the model used a particular cue.

Include apparently useful earlier context that the model fails to exploit, and cases where a short-context method is better. Select cases by a reproducible category-based rule rather than only taking the largest neural improvements.

## Expected Research Artifacts

- A precise definition and count table for each novelty stratum.
- Full-test and known-word-unseen-combination performance tables.
- Loss comparisons within target-frequency bands.
- The short-versus-long-context ablation with seed variability and runtime.
- At least ten sequence analyses retaining original and mapped text.
- If synthetic data is used, a generator specification and template-disjointness audit.

## Interpretation, Risks, and Limits

If longer context helps, the evidence supports the usefulness of additional history under this protocol. It does not isolate representation sharing from all optimization effects, and it does not prove systematic compositionality.

If smoothing performs as well as the longer-context model, the selected corpus may contain sufficient local information, the longer-context model may be poorly estimated, or the diagnostic may not require longer dependencies. Use validation evidence and the error analysis to distinguish plausible explanations without retroactively redefining success.

Unseen n-grams occur naturally even in unremarkable sentences. Conversely, a seen trigram can occur in a semantically difficult context. Novelty counts are an operational diagnostic, not a complete account of linguistic complexity.

Other limitations include corpus-specific vocabulary, sparse diagnostic strata, residual frequency confounding, window-training cost, and subjectivity in qualitative cue interpretation. Synthetic-only findings must be explicitly restricted to the generator's language.

## Fit with PS-1

The design includes all four required model families, one held-out test set, required metrics, a controlled ablation, and at least ten rare/unseen sequence analyses. It directly addresses sparse and unseen word sequences rather than introducing a classification or parsing task.

Use reproducible Python dependencies and record seeds, data version or checksum, source/access date, terms, sizes, preprocessing, split manifests, and limitations. NLTK lists Brown for non-commercial use. Declare adapted code, AI assistance, and all pretrained resources if introduced. Use three stochastic seeds where feasible or justify a single run.

## Literature Starting Points

1. Bengio et al. (2003), *A Neural Probabilistic Language Model*. Shared representations and unseen sequences.
2. Chen and Goodman (1999), *An Empirical Study of Smoothing Techniques for Language Modeling*. Sparse-event probability estimation.
3. Hochreiter and Schmidhuber (1997), *Long Short-Term Memory*. Recurrent context processing.
4. Mikolov et al. (2010), *Recurrent Neural Network Based Language Model*. Neural language-model evaluation.
5. Lake and Baroni (2018), *Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks*. Motivation for careful generalization tests; its task differs from next-word modelling.
6. Keysers et al. (2020), *Measuring Compositional Generalization: A Comprehensive Method on Realistic Data*. Split design and operationalizing compositional novelty; not direct proof about this project's models.

Read and verify the references before citing, and distinguish evidence from other tasks from evidence about next-word prediction. Dataset/tool documentation is additional: [NLTK corpus access](https://www.nltk.org/howto/corpus.html), [data terms](https://www.nltk.org/nltk_data/), and [language-model examples](https://www.nltk.org/howto/lm.html).