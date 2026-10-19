# Direction 4: Language-Model Robustness under Genre Shift

## Research Focus

**Working title:** When the Writing Style Changes: Statistical and Neural Language Models under Genre Shift and Limited Adaptation.

This direction investigates what happens when language models trained on one writing genre are used on another, and whether a small amount of target-genre text can improve prediction without destroying source-genre performance.

The task remains next-word language modelling. It is not PS-2's POS-tagging task: the models predict words, and the central measurements are perplexity, vocabulary, OOV rate, and runtime.

This is a proposed research design. No performance gains, best methods, or novel contributions are established in advance.

## Problem and Motivation

A language model trained on news text may learn typical reporting phrases, named entities, and sentence patterns. Fiction may contain different dialogue patterns, descriptions, vocabulary, and continuation probabilities.

Even a model that performs well on held-out news can therefore perform poorly on fiction. However, a higher target-domain perplexity can have several explanations: unfamiliar vocabulary, changed word frequencies, changed local combinations, longer-range differences, or simply a harder evaluation population.

The research problem is to measure this failure carefully and test a bounded adaptation method. The intervention should not be confused with merely giving one model more data than another.

## Research Questions and Hypotheses

**Primary question:** Can a small, fixed amount of target-genre training text improve target prediction, and what source-genre performance is lost or preserved?

**Secondary questions:**
- Which required language-model family shows the largest performance change between source and target test sets?
- How much of the observed difference involves OOV-containing contexts versus combinations of known words?
- Does mixing source examples into neural adaptation preserve source performance better than target-only fine-tuning?
- Does a simple adapted statistical model remain competitive at the same target-data budget?

**Testable hypotheses:**
- H1: Models trained only on source text perform differently on held-out source and target text, with increased target loss expected but not guaranteed.
- H2: Limited target adaptation reduces target loss relative to the same model before adaptation.
- H3: Source replay during neural adaptation reduces source-performance degradation relative to target-only adaptation.

H3 does not imply that replay must achieve the lowest target perplexity. The research concerns the trade-off between target improvement and source retention.

## Data Design

A feasible starting point is a source drawn from Brown `news` and `editorial`, with `fiction` as target. These are proposed choices; inspect available document counts and coordinate the variant with Groups 6 and 11 before finalizing.

Use the same corpus collection for both genres to reduce differences caused by incompatible tokenization pipelines. Brown's genre labels identify the evaluation domains, not the next-word prediction targets. No manual word labels are required.

Construct separate document partitions:

| Partition | Permitted use |
|---|---|
| Source training | Train all source-only models and supply replay text. |
| Source validation | Select source models and monitor source retention during adaptation. |
| Source test | Final measurement of in-domain quality and forgetting. |
| Target adaptation training | The only target text allowed to update model parameters or counts. |
| Target validation | Select adaptation settings and stopping points. |
| Target test | Final target-domain evaluation only. |

If the target genre is small, use conservative document allocations and report limited evaluation size. Keep all source and target tests fixed. Audit exact and near-duplicate sentences/documents across all partitions, not just within each genre, and document removals or grouping decisions.

### Vocabulary and OOV Control

Build the primary vocabulary from source training only. Replace rare source words with `<UNK>` so the token is represented in training, and apply the same mapping to every target partition and every model variant.

This keeps pre-adaptation and post-adaptation perplexities on the same output space. It also imposes a limitation: adaptation cannot introduce new lexical outputs. The experiment primarily measures changes in predictions over the existing vocabulary and `<UNK>`.

Report source and target OOV rates before replacement. They need not be equal across domains, but must be equal across model variants within a domain. Distinguish target-word OOV from histories containing OOV words.

A higher target OOV rate can sometimes make mapped text artificially easier because many distinct words collapse into `<UNK>`. Consequently, domain perplexity differences alone do not quantify linguistic shift. Report known-target or known-context diagnostics and explain that these subsets are selected, not representative replacements for the complete test set.

Vocabulary expansion is an optional separate study. Expanding the output space during adaptation changes the scoring task and must not be mixed into the primary fixed-vocabulary comparison.

## Required Baselines

Train the following on source training text only:

| Model | Purpose |
|---|---|
| Unsmoothed bigram | Required baseline and local sparsity reference. |
| Unsmoothed trigram | Required higher-order sparsity reference. |
| Smoothed trigram | Required robust statistical comparison. |
| Small word-level LSTM | Required neural comparison, trained from scratch. |

Evaluate every baseline on both source and target held-out text before adaptation. The same test sets are shared across methods within each domain. Keep unadapted model copies for paired pre/post comparisons.

An add-k trigram is a transparent starting smoother. A normalized interpolated smoother is a useful stronger alternative if resources permit. Hyperparameters are selected on source validation data for source-only models.

## Proposed Intervention: Source Replay

**Target-only neural adaptation:** continue training the source LSTM using only the limited target-adaptation text.

**Replay adaptation:** continue training an identical copy using a mixture of target-adaptation text and examples sampled from source training. The purpose of replay is to preserve source patterns while learning target patterns.

The adapted models must begin from the same source checkpoint for each seed. Source replay may only use source training documents, never source validation or source test text. Report the replay fraction, sampling rule, learning rate, number of updates, target/source tokens processed, and stopping criterion.

This is a test of a familiar adaptation principle in a constrained language-modelling setting, not a claim that rehearsal or replay is new.

### A Statistical Adaptation Comparator

Fit a smoothed target trigram on the target-adaptation partition using the same fixed vocabulary. Combine it with the source smoothed trigram using a validation-selected constant probability mixture. This provides a lightweight adaptation comparison using the same target data.

Include a target-only smoothed trigram if feasible. It helps distinguish the value of retaining source information from simply modelling the new genre directly. There is no need to add several neural architectures or adaptation techniques for the main question.

### Exposure and Compute Controls

Replay creates an unavoidable experimental choice: matching target exposure generally adds source computation, while matching total updates generally reduces target exposure.

For the primary comparison, match target-token exposure between target-only adaptation and replay, and explicitly report the extra replay computation. Include an equal-total-compute target-only control if feasible, allowing it to revisit the same target training data. This tests whether replay's benefit is explained merely by more optimization.

If the compute control cannot be run, limit the conclusion: the experiment tests the combined effect of replay and its additional computation, not a fully isolated advantage at equal cost.

## Experimental Protocol

1. Freeze all six data partitions, duplicate handling, vocabulary, preprocessing, and target adaptation budget.
2. Train source-only required models and select their settings using source validation data.
3. Create matched copies of each source neural checkpoint for target-only and replay adaptation.
4. Train the target statistical comparator using exactly the permitted adaptation text.
5. Select adaptation settings on validation data using a declared rule, such as lowest target validation loss subject to a specified source-loss tolerance.
6. Freeze weights, replay ratios, stopping points, and analysis rules before final testing.
7. Evaluate every variant on both fixed test sets and report paired changes from its unadapted baseline.
8. Use at least three neural seeds where feasible, with matched starting checkpoints across adaptation methods. Report mean and standard deviation or justify fewer runs.

The source-retention tolerance must be chosen before seeing test outcomes. Alternatively, report a small predeclared set of replay ratios as a trade-off curve rather than selecting one after test inspection.

Start with one target-adaptation budget. A second budget can test sensitivity to target-data availability, but a large budget sweep is not necessary to establish the main result.

## Metrics and Required Ablations

Report perplexity, vocabulary size, OOV rate, and runtime on both domains. Separate source training, adaptation, and inference time, and record hardware and device settings.

Compute perplexity from total token negative log-likelihood with identical scoring conventions: actual words plus one sentence-ending target, excluding beginning-of-sentence context and batch padding. Unsmoothed full-test perplexity may be infinite; retain infinity and report zero-probability event rates rather than inserting an epsilon.

Measure source retention using the change in source-test negative log-likelihood from before to after adaptation. Measure target improvement using the corresponding change on the target test. Reporting loss differences as well as perplexity makes trade-offs easier to interpret.

**Primary ablation:** remove source replay while keeping the source checkpoint, target data, architecture, and declared exposure policy unchanged.

**Supporting smoothing comparison:** unsmoothed versus smoothed trigram in both domains. This indicates how much a basic sparsity treatment helps without target adaptation.

Known-word and OOV-containing strata provide supporting diagnostics. They are not a causal decomposition of genre shift, because genre, vocabulary, syntax, and sentence difficulty remain correlated.

## Qualitative Error Analysis

Analyse at least ten rare/unseen held-out sequences spanning both domains. Include target examples improved by adaptation, target examples still poorly predicted, source examples harmed by target-only adaptation, and source examples preserved or worsened by replay.

For each case, show the prefix, actual next word, source and target-adaptation counts, OOV status, pre/post model probabilities, and a cautious interpretation. Distinguish lexical novelty from an unfamiliar combination of known words.

Do not use source or target test cases to refine the replay rule. Example selection should follow declared categories and include failures of the proposed method, not only successful adaptation.

## Expected Research Artifacts

- Source/target dataset tables, OOV statistics, and partition audit.
- Source-only model results on both test domains.
- Pre/post adaptation results for target-only neural adaptation, replay, and statistical adaptation.
- Source-retention versus target-improvement plots with neural seed variability.
- Replay ablation and adaptation-runtime measurements.
- At least ten annotated rare/unseen sequence cases.

## Interpretation, Risks, and Limits

If replay preserves source quality but gives less target improvement, report the trade-off rather than declaring an unconditional winner. If target-only adaptation improves both domains, the source model may have been undertrained or the genres may share useful patterns; forgetting is not guaranteed.

If replay adds no benefit over an equal-compute control, the evidence does not support the proposed mechanism at the chosen scale. If a lightweight statistical adaptation method is competitive, that can be the practical conclusion.

Source and target test sets contain different sentences. Their raw perplexity difference alone does not establish the causal effect of domain shift. Stronger evidence comes from comparing adapted and unadapted models on the identical target test and monitoring changes on the identical source test.

Other limits include one source-target pair, a fixed source vocabulary, modest target data, validation-selection uncertainty, and potentially different training exposure. Brown's historical written genres do not establish robustness to modern chat, speech, or technical text.

## Fit with PS-1

The experiment retains all required language models and evaluates sparse/unseen word prediction with the required metrics, held-out data, ablation, and at least ten sequence analyses. Domain shift is an experimental condition within language modelling, not a change to the assigned task.

Use reproducible Python dependencies, fixed split manifests, recorded seeds, and documented corpus source/version or checksum, access date, terms, sizes, preprocessing, hardware, and limitations. NLTK lists Brown for non-commercial use. Declare borrowed code and AI assistance. If pretrained components are used, document exact identity/version, licence, source, download date, and whether they are frozen or fine-tuned.

## Literature Starting Points

1. Chen and Goodman (1999), *An Empirical Study of Smoothing Techniques for Language Modeling*. Statistical robustness to sparse events.
2. Bengio et al. (2003), *A Neural Probabilistic Language Model*. Neural next-word prediction.
3. Mikolov et al. (2010), *Recurrent Neural Network Based Language Model*. Recurrent language modelling.
4. Moore and Lewis (2010), *Intelligent Selection of Language Model Training Data*. Domain-relevant language-model data selection.
5. Gururangan et al. (2020), *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*. Domain adaptation motivation; its large pretrained-model setting differs from this small-model study.
6. Robins (1995), *Catastrophic Forgetting, Rehearsal and Pseudorehearsal*. Background for the replay hypothesis, not direct evidence of a benefit in this corpus.

Read the papers, verify citation details, and distinguish results from different model scales and tasks. Dataset/tool documentation is additional: [NLTK corpus access](https://www.nltk.org/howto/corpus.html), [data terms](https://www.nltk.org/nltk_data/), and [language-model examples](https://www.nltk.org/howto/lm.html).