# Direction 1: Data Efficiency of Statistical and Neural Language Models

## Research Focus

**Working title:** How Much Data Is Enough? A Controlled Study of Statistical and Neural Language Models under Limited Training Data.

This direction investigates how the amount of available training text changes the relative strengths of counting-based and neural language models. The research target is not a single leaderboard. It is the relationship between data availability, prediction quality, sparse-context failures, and computational cost.

The central question is whether there is a region in which a smoothed n-gram is preferable to a small neural model, and whether that preference changes as more data becomes available. A neural advantage, a statistical advantage throughout the tested range, and the absence of a clear crossover are all possible outcomes.

This document proposes an experiment. It contains no measured results and does not establish that the proposed findings would be novel.

## Problem and Motivation

An unsmoothed n-gram estimates next-word probabilities from observed counts. With little text, many valid sequences have no observations. Increasing the context from one word to two words can make this problem worse because there are more possible contexts to estimate.

Smoothing assigns probability to combinations that were not observed. A neural model instead learns shared parameters and representations, potentially transferring information between contexts. However, parameter sharing does not guarantee good generalization: a small training set may not support the neural model's capacity, and optimization or overfitting can dominate its results.

The practical issue is therefore not simply whether neural models can outperform n-grams. It is whether their extra training cost is justified at the data scale actually available.

## Research Questions and Hypotheses

**Primary question:** How does training-data availability affect held-out perplexity and computational cost for unsmoothed n-grams, a smoothed trigram, and a small LSTM?

**Secondary questions:**
- Does the preferred model change over the observed data range?
- Which frequency groups benefit most from additional training text?
- Does smoothing provide its largest benefit when training data is scarce?
- Does a model's aggregate improvement hide continued failure on rare or unseen contexts?

**Testable hypotheses, not assumed conclusions:**
- H1: At the smallest training budgets, the smoothed trigram is competitive with the LSTM.
- H2: The LSTM's relative performance improves as training data grows.
- H3: Additional data affects frequent and sparse contexts differently.

H2 is not supported if the neural model does not improve relative to the statistical model over the tested range. If no crossover is observed, report that result rather than extrapolating a crossover beyond the available evidence.

## Data and Evaluation Population

Use one documented Brown Corpus genre subset, such as press text drawn from `news`, `editorial`, and `reviews`. Confirm distinctness from Groups 6 and 11 before fixing the selection. WikiText-2 is an alternative, but one corpus is sufficient for the main experiment.

No manual class labels are needed. The next word in each sentence is the prediction target. Preserve sentence order and document identifiers; do not use Brown's POS tags as model inputs.

Create disjoint training-pool, validation, and test document sets. A proposed starting allocation is approximately 80/10/10 by document, with genre representation preserved. Audit normalized exact duplicates and near-duplicate sentences or documents before freezing the split. Document the similarity rule, adjudication, and removals. Shared individual words and short common phrases do not by themselves constitute duplicate documents.

Construct nested training subsets from the training pool, for example 10%, 25%, 50%, and 100%. Select whole documents in a seeded, genre-aware order. Report actual token counts because document percentages do not imply exact token percentages. Keep validation and test documents unchanged across budgets.

### Controlling Vocabulary across Training Budgets

Changing the output vocabulary changes the prediction task, so perplexities from different vocabulary policies are not directly interchangeable.

For the primary controlled experiment, build a fixed vocabulary from the smallest training subset only. Replace its singleton words with `<UNK>` so that unknown-token behaviour is learned, then apply this vocabulary to every larger training subset and to validation/test text. Use the same predicted token set for all models and budgets.

This isolates the effect of additional observations under a fixed lexical inventory. Its limitation is important: larger datasets cannot expand the vocabulary, so the experiment does not measure the full practical benefit of obtaining more text. Report the resulting OOV rate and avoid aggressive cutoffs that collapse much of the corpus into `<UNK>`.

An optional secondary experiment can rebuild vocabulary at each budget. Report it separately as an end-to-end data-growth experiment, not as a directly comparable fixed-vocabulary perplexity curve. Within each budget, all models must still share one vocabulary.

## Systems and Intervention

| System | Role |
|---|---|
| Unsmoothed bigram | Required baseline using one previous word. |
| Unsmoothed trigram | Required baseline using two previous words. |
| Smoothed trigram | Required smoothing comparison; use add-k initially or a documented normalized interpolated smoother. |
| Small word-level LSTM | Required neural comparison, trained from scratch. |

The intervention is controlled access to additional training text, not a new architecture. This is an empirical characterization study rather than a claim of algorithmic novelty.

Use a modest, fixed neural architecture across budgets. A one-layer LSTM with 128-dimensional embeddings and 256 hidden units is a starting configuration, not a required or proven-optimal choice. Keep the optimizer, maximum training policy, and early-stopping rule documented.

Choose smoothing and neural hyperparameters using validation data only. Use the same candidate search space and tuning allowance at each budget. This estimates performance under a consistent model-selection procedure. A secondary fixed-hyperparameter analysis can isolate data growth more strictly, but is not necessary for the minimum study.

## Experimental Protocol

1. Freeze document partitions, duplicate handling, preprocessing, vocabulary, and the nested training-subset order.
2. Define frequency groups from training counts, retaining original-word OOV flags before `<UNK>` replacement.
3. Fit every required model at every training budget.
4. Select hyperparameters and neural stopping checkpoints using validation data.
5. Run at least three neural initialization/training seeds where feasible, holding data subsets fixed. Report mean and standard deviation; justify fewer runs if necessary.
6. Freeze the evaluation procedure before inspecting final test performance. Run the predefined test matrix and retain all results, including unfavourable ones.
7. Compare overall performance, frequency-conditioned losses, zero-probability events, and measured runtime.

Initialization seeds measure neural training variability, not uncertainty due to which documents were sampled. If resources allow, repeat the nested document sampling as a separate experiment. Do not present three initializations as three independently sampled datasets.

Train to the same validation-based stopping rule with a documented maximum budget, rather than assuming equal epoch counts imply equal optimization quality. Report actual epochs, updates, tokens processed, and elapsed time. An equal-compute experiment is optional and answers a different question from data efficiency.

## Metrics and Analysis

Required outputs are held-out perplexity, vocabulary size, OOV rate, and runtime. Split runtime into fitting/training and test scoring, and record hardware, device, batch size, and timing boundaries.

Compute corpus perplexity from total negative log probability divided by the number of scored tokens, then exponentiate. Do not average sentence perplexities. Score identical target words and one sentence-ending token across models; exclude beginning-of-sentence context and batch padding from the denominator.

If an unsmoothed model assigns zero probability to any evaluated event, its full-test perplexity is infinite. Report that honestly, together with the fraction of zero-probability events. Do not replace zeros with small constants while still calling the model unsmoothed.

Useful supplementary measurements include negative log-likelihood by context-frequency bucket, throughput, and the gap between validation and training loss. Define rare and unseen groups before final evaluation. For cross-budget comparisons, distinguish fixed groups defined at the smallest budget from groups recomputed at each budget; they answer different questions.

## Required Ablation

Compare the unsmoothed trigram with the smoothed trigram at the same training budget, using identical tokens, vocabulary, and contexts. This isolates smoothing.

The data-budget curve is itself a controlled comparison, but retaining an explicit smoothing ablation makes the assignment requirement unambiguous. If a stronger smoother is added, compare it against add-k under the same conditions rather than adding many unconnected variants.

## Qualitative Error Analysis

Analyse at least ten held-out rare or unseen sequences. Include events that become observed as training data increases, events that remain sparse, unseen combinations of familiar words, and OOV-containing events.

For each case, show the original context, target word, mapped representation, training counts at each budget, model probabilities, and a short explanation. Include cases where extra training data fails to help or where the neural model becomes worse.

Select examples using declared categories and a reproducible selection rule. Do not choose only examples supporting a crossover. Predicting `<UNK>` is not evidence that a model can predict the original unseen lexical item.

## Expected Research Artifacts

- Dataset statistics and training-budget tables with exact document and token counts.
- Learning curves of test perplexity against training tokens, with neural seed variability shown.
- Runtime-versus-quality plots, with infinite baseline perplexities explicitly annotated.
- Smoothing-ablation results at matched budgets.
- Frequency-conditioned loss and zero-probability summaries.
- At least ten sequence-level analyses, including counterexamples.

These are proposed outputs, not expected numerical outcomes.

## Interpretation, Risks, and Limits

A crossover would support a conditional recommendation tied to this corpus, vocabulary, architecture, and training protocol. It would not establish a universal data threshold for neural language modelling.

If the neural model never wins, investigate optimization, overfitting, and model capacity using validation evidence before interpreting the result. Do not redesign the system repeatedly in response to final test scores. Conversely, an early neural advantage does not prove that smoothing is unnecessary.

Main limitations include the fixed restricted vocabulary, one corpus selection, a small number of neural seeds, training-subset sampling variability, and hardware-specific timing. Natural sentences within documents are dependent; token-level resampling can overstate precision. Optional paired document bootstrap intervals should be labelled exploratory and distinguished from seed variability.

## Fit with PS-1

The design includes all four required models, a common held-out test set, perplexity/vocabulary/OOV/runtime reporting, an explicit smoothing ablation, and at least ten rare/unseen sequence analyses. Use Python with pinned dependencies and record seeds, configurations, split manifests, and actual execution conditions.

Document the corpus source, access date, terms, exact version or archive checksum, preprocessing, sizes, and limitations. NLTK lists Brown for non-commercial use; do not call it public domain. Declare borrowed code and generative-AI assistance. If pretrained components are introduced, record their exact identity, version, source, licence, download date, and frozen/fine-tuned status.

## Literature Starting Points

Read these as starting points, verify publication details, and explain how each informs the experiment. A literature search is still needed before claiming a research gap.

1. Chen and Goodman (1999), *An Empirical Study of Smoothing Techniques for Language Modeling*. Smoothing choices and controlled evaluation.
2. Kneser and Ney (1995), *Improved Backing-Off for M-Gram Language Modeling*. A stronger statistical comparison.
3. Bengio et al. (2003), *A Neural Probabilistic Language Model*. Parameter sharing and neural next-word prediction.
4. Hochreiter and Schmidhuber (1997), *Long Short-Term Memory*. The recurrent architecture.
5. Mikolov et al. (2010), *Recurrent Neural Network Based Language Model*. Recurrent language modelling and statistical comparisons.
6. Goodman (2001), *A Bit of Progress in Language Modeling*. Broader empirical language-modelling methodology.

Dataset/tool documentation, additional to scholarly references: [NLTK corpus access](https://www.nltk.org/howto/corpus.html), [NLTK data terms](https://www.nltk.org/nltk_data/), and [NLTK language-model examples](https://www.nltk.org/howto/lm.html).