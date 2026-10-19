# Direction 2: Selective Combination of Statistical and Neural Language Models

## Research Focus

**Working title:** When Should a Language Model Trust Counts? Context-Aware Combination of Statistical and Neural Predictors.

This direction investigates whether a simple, observable signal can indicate when to rely more heavily on a smoothed trigram or a small neural language model. The target contribution is a controlled study of model complementarity and a lightweight selection mechanism, not the invention of model interpolation.

The proposed signal is the frequency of the preceding two-word context in training data. The experiment asks whether that signal improves on a properly tuned constant mixture.

This document describes a proposed study. No results have been measured, and novelty must be assessed through a literature review.

## Problem and Motivation

A trigram estimates the next word from a two-word history. A frequently observed history may have useful empirical continuation counts. A rare history supplies less direct evidence, and an unseen history depends entirely on the smoothing or backoff mechanism.

A neural model can share learned information across histories, but this does not make it consistently superior. It may underfit a frequent expression, overfit limited text, or distribute probability poorly despite producing plausible continuations.

A fixed combination can exploit these differences, but uses the same weight everywhere. The specific limitation to investigate is whether that constant weight ignores predictable changes in relative model reliability.

Importantly, context frequency is only a proxy. A common context can have many possible continuations, while a rare context can be easy. The study must test this proxy rather than assume that frequency equals certainty.

## Research Questions and Hypotheses

**Primary question:** Does training-context-frequency-dependent interpolation reduce held-out perplexity compared with each component and with the best validation-selected constant interpolation?

**Secondary questions:**
- Do the statistical and neural models have different error patterns across context-frequency groups?
- Is any gain concentrated in rare contexts, or does it come from common ones?
- Does the adaptive rule generalize beyond the validation data used to select its weights?
- Is the improvement large enough to justify evaluating two models at inference time?

**Testable hypotheses:**
- H1: Relative statistical-versus-neural prediction loss varies with training-context frequency.
- H2: A low-complexity context-dependent mixture improves over a constant mixture on held-out data.
- H3: The adaptive mixture's advantage is not explained solely by one unusually favourable neural seed.

H1 can hold without H2: complementary behaviour may exist but be too weak or unstable for the proposed gate to exploit.

## Data and Shared Prediction Task

Use one documented natural-text corpus subset, such as Brown press text. Coordinate the selected genres or dataset variant with Groups 6 and 11. No manual labels are required because the next word is the target.

Split whole documents into training, validation, and final test partitions, with genre representation maintained where relevant. Audit exact and near-duplicate sentences/documents and resolve cross-partition overlap before freezing the split. Record the normalization and similarity rules used by the audit.

Build a vocabulary from training data only. A possible initial rule replaces training singletons with `<UNK>`, then maps validation/test words outside the vocabulary to `<UNK>`. Keep original tokens and OOV flags for analysis. Both models and all mixtures must predict the same vocabulary with compatible sentence boundaries.

Use one casing policy and Brown's existing tokenization if Brown is chosen. Preserve stopwords and word order. Do not turn the task into prediction over stemmed content words without explicitly redefining the research question.

## Baselines and Proposed Method

| System | Purpose |
|---|---|
| Unsmoothed bigram | Required baseline. |
| Unsmoothed trigram | Required higher-order baseline. |
| Smoothed trigram | Statistical component and required smoothing comparison. |
| Small word-level LSTM | Neural component and required neural comparison. |
| Fixed-weight mixture | Strong baseline for the benefit of combining components. |
| Frequency-dependent mixture | Proposed intervention. |

Start with add-k smoothing for a transparent implementation. A normalized interpolated smoother such as interpolated Kneser-Ney is a valuable stronger statistical check if feasible. A claim about the adaptive mechanism is weaker if it only compensates for an unnecessarily poor statistical component.

Use a small LSTM trained from scratch. Freeze the component models after validation-based model selection; the mixture experiment should not secretly give its components additional training data.

### Probability Mixture

For a next word w and available history h:

`P_mix(w | h) = lambda(h) * P_trigram(w | h) + (1 - lambda(h)) * P_LSTM(w | h)`

The trigram uses the last two words of h; the LSTM may use a longer documented history. The mixing weight lies between zero and one and must be the same across candidate words for a given history. Combining normalized distributions in this way preserves normalization.

For the fixed mixture, use one constant weight selected on validation data. Include weights zero and one among the candidates so that the search can recover either component.

For the proposed mixture, define three history-frequency buckets using training counts: unseen, rare, and frequent. Set the rare/frequent threshold in advance or choose it through a small documented validation search. Assign one validation-selected mixing weight to each bucket.

Keep the gate deliberately small. A three-weight rule is more interpretable and easier to test than an additional neural gating network. Decide before final testing whether weights are unconstrained or constrained to increase statistical reliance with frequency; the latter encodes a stronger assumption.

### Information Available to the Gate

The gate may use the history and its training frequency. It must not use the actual next word, the probability assigned to the actual next word, or whether the prediction turned out correct. Those quantities are only known after the event and would leak the target.

Define whether history counts are taken before or after `<UNK>` mapping. Using the mapped history is operationally consistent with the component models, but can merge unrelated rare contexts. Preserve raw-context counts for diagnostic analysis and state this limitation.

## Experimental Protocol

1. Freeze data partitions, vocabulary, boundary handling, frequency definitions, and metric implementation.
2. Train the required components and select their hyperparameters on validation data.
3. Inspect validation loss differences across context-frequency groups to test whether the proposed signal has any visible relationship with model reliability.
4. Tune the fixed weight and the adaptive bucket weights on validation predictions using total negative log-likelihood.
5. Freeze every weight, threshold, checkpoint, and analysis rule before final test evaluation.
6. Evaluate all systems on the identical test target stream, including examples where the mixture hurts.
7. Repeat neural training for at least three seeds where feasible and select mixture parameters separately on validation predictions for each corresponding run. Report this procedure explicitly.

Repeatedly inspecting validation results can also overfit model development. Limit the gate's complexity and search space. If enough validation documents are available, a separate gate-development and gate-check partition can diagnose this risk; it is not necessary to add a large tuning infrastructure.

Mixture tuning can reuse cached component probabilities for the observed validation targets. Full vocabulary distributions are only necessary for normalization checks, top-prediction examples, or generation. Reusing predictions reduces experiment cost without changing the statistical question.

## Metrics and Comparisons

Report the required test perplexity, vocabulary size, OOV rate, and runtime. Compute perplexity from summed token negative log probabilities, not from an average of sentence perplexities or a weighted average of component perplexities.

All systems score the same words and one sentence-ending target, excluding beginning-of-sentence context and padding. Vocabulary size and OOV rate should be identical across systems under this design; they describe the shared task rather than a mixture advantage.

Report training cost, mixture-selection cost, and actual inference cost separately. A probability mixture generally requires both component predictions, even when its gate is inexpensive. Cached experimental scoring time is not a substitute for end-to-end inference timing.

Useful supplementary results include loss by context-frequency bucket, learned weights, per-seed improvement over fixed interpolation, and the number of events improved or harmed. Event win rate alone is insufficient: a few large losses can outweigh many small gains.

Unsmoothed baselines may have infinite full-test perplexity. Retain that result and show their zero-probability event rate. Do not smooth them implicitly with an epsilon. Use normalized component probabilities; an unnormalized backoff score cannot be inserted into a probability mixture and interpreted as ordinary perplexity without further work.

## Required Ablations

**Primary ablation:** replace the context-dependent weights with one validation-selected constant weight while keeping the component models and evaluation stream unchanged. This directly tests the value of context dependence.

**Assignment-aligned supporting comparison:** compare unsmoothed and smoothed trigrams with identical data, vocabulary, and context order.

An optional stricter check compares the adaptive gate with a simpler two-bucket gate. Add it only if it answers a concrete concern about overfitting; many gate variants can obscure rather than strengthen the central claim.

## Qualitative Error Analysis

Analyse at least ten actual rare/unseen test sequences. Include cases where the statistical model is better, the neural model is better, the gate chooses a helpful weight, and the gate makes the combined prediction worse.

For each example, show history, true next word, training history count, frequency bucket, selected weight, component probabilities, mixture probability, and interpretation. Preserve the original word alongside `<UNK>` where mapping occurs.

A useful counterexample is a frequent but ambiguous history for which the statistical model is unreliable. Another is a rare fixed expression that the counting model handles better than the neural model. These probe the mechanism rather than merely displaying successful outputs.

## Expected Research Artifacts

- A six-system comparison table with required metrics and neural seed summaries.
- A table of fixed and adaptive weights selected without test access.
- Loss-difference plots by history-frequency bucket.
- The adaptive-versus-fixed ablation and smoothing comparison.
- End-to-end inference cost measurements.
- At least ten annotated sequence-level cases.

## Interpretation, Risks, and Limits

If adaptive interpolation beats the components but not fixed interpolation, the evidence supports model combination but not the proposed context signal. This is a useful negative result about the intervention.

If gains appear only on validation data, likely explanations include gate overfitting or unstable bucket estimates. If results vary substantially across neural seeds, report that instability rather than selecting the best run.

Potential confounds include weak component tuning, excessive `<UNK>` mapping, context frequency acting as a proxy for genre, and different context lengths across component models. The primary ablation isolates the gate, not the individual architectural reasons for complementarity.

Interpolation and adaptive interpolation have substantial prior literature. The defensible contribution is a specific, reproducible evaluation of this signal under the selected data conditions. Do not claim a new family of language models without establishing that claim through literature.

## Fit with PS-1

The design retains the required unsmoothed bigram, trigram, smoothed/backoff model, and small neural LM. It uses a common test set, required metrics, an explicit ablation, and at least ten rare/unseen examples. Three stochastic seeds are preferred where feasible, with mean and standard deviation or a documented justification for fewer runs.

Record Python dependencies, seeds, splits, corpus source/version or checksum, access date, terms, preprocessing, vocabulary policy, runtime conditions, and limitations. NLTK lists Brown for non-commercial use. Declare adapted code and AI assistance, and fully document any pretrained component if one is introduced.

## Literature Starting Points

1. Jelinek and Mercer (1980), *Interpolated Estimation of Markov Source Parameters from Sparse Data*. Foundational interpolation ideas.
2. Chen and Goodman (1999), *An Empirical Study of Smoothing Techniques for Language Modeling*. Statistical component quality.
3. Kneser and Ney (1995), *Improved Backing-Off for M-Gram Language Modeling*. A stronger smoothing option.
4. Bengio et al. (2003), *A Neural Probabilistic Language Model*. Shared representations for next-word prediction.
5. Mikolov et al. (2010), *Recurrent Neural Network Based Language Model*. Recurrent models and statistical-neural comparisons.
6. Goodman (2001), *A Bit of Progress in Language Modeling*. Combining language-modelling techniques and evaluating gains.

Read the papers and verify bibliographic details before citing. Search specifically for adaptive interpolation and context-dependent language-model combination before asserting novelty. Dataset/tool documentation is additional: [NLTK corpus access](https://www.nltk.org/howto/corpus.html), [data terms](https://www.nltk.org/nltk_data/), and [language-model examples](https://www.nltk.org/howto/lm.html).