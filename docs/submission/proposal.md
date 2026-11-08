# Retrospective Proposal: Smoothing, Sparsity and Vocabulary Growth

**Group 1, Problem Statement 1. Prepared 7 November 2026 after the study; not a contemporaneous preregistration.**

| Author | Student ID |
|---|---|
| Akshaya Kumar Joish | 2025cs05013 |
| Anuj Kumar Srivastava | 2025cs05040 |
| Murali Das Mohan | 2025cs05052 |

**Research question.** How do smoothing and neural parameter sharing affect next-word prediction under sparse observations, and how do vocabulary growth and unseen-context categories explain model rankings across training budgets?

**Datasets.** Brown, Gutenberg, Inaugural, Reuters, ROCStories and WebText provide six distinct English text domains. Fixed document partitions supply 20/40/60/80/100% prefixes of sorted training sentence lines, not random samples or whole-document budgets. WebText uses one training, two validation and two test source files under split seed 13. Each prefix rebuilds its vocabulary: Brown minimum frequency 2, other corpora 1. Lowercase whitespace tokenization is shared.

**Baselines and advanced models.** Compare bigram/trigram MLE, add-one and absolute-discount backoff with FFNNs using one/two-token histories and the approved sentence-prefix LSTM, resetting per sentence. Neural models use embeddings 128, hidden units 256, dropout 0.3, Adam 0.001, clipping 1, and at most 20 epochs with strict validation-NLL selection and patience 3. The completed matrix contains 240 neural fits and 180 statistical conditions, giving 420 test conditions. Brown, Gutenberg, Inaugural, Reuters and WebText use seeds 13/42/2026; ROCStories uses seed 13 only.

**Metrics and ablations.** Report EOS-inclusive perplexity, word-only accuracy/OOV, runtime and seed summaries. Compare smoothing, matched FFNN history lengths and ordered data budgets; saved-event analyses separate unknown targets, unseen mapped contexts and a fixed retained-target cohort. These comparisons do not isolate causal data-size or LSTM context-length effects.

**Risks and scope.** Changing vocabularies, unequal compute, limited held-out sources, ROCStories single-seed uncertainty and retained cross-split matches restrict inference. Brown, Gutenberg, Inaugural, Reuters and ROCStories report zero exact overlap after quarantine; the final WebText split retains 37 exact strings across splits. ROCStories and WebText were not near-scanned. The group reported instructor approval for supplementary corpora and disclosed retained-overlap flags on 7 November 2026; no written record was supplied. Approval does not establish independence. The 10 MB submission excludes raw data, checkpoints and large events, limiting replay.

**Contributions:** everyone contributed equally.

Protocol and limitations: [dataset card](dataset_card.md), [reproducibility](reproducibility.md).
