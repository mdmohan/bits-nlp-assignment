# Smoothing, Sparsity and Vocabulary Growth: Comparing Backoff and Neural Language Models Across Training Budgets

**Group:** 1

| Author | Student ID |
|---|---|
| Akshaya Kumar Joish | 2025cs05013 |
| Anuj Kumar Srivastava | 2025cs05040 |
| Murali Das Mohan | 2025cs05052 |

**Contributions:** All three authors contributed equally.

## Abstract

Sparse observations cause unsmoothed n-gram language models to assign zero probability to plausible unseen sequences. We compare bigram and trigram maximum-likelihood, add-one and absolute-discount backoff models with context-matched feed-forward neural models and a sentence-level LSTM. The models are evaluated across six corpora and five training budgets using perplexity, accuracy, OOV rate, runtime and event-level error categories. Unsmoothed models consistently produce infinite held-out perplexity, while backoff provides the strongest aggregate performance on four corpora and LSTM on two. Neural gains are concentrated in particular unseen-context conditions rather than occurring universally, and mapped <unk> predictions can substantially inflate apparent accuracy. We conclude that smoothing reliably addresses zero-probability failure, but the relative advantage of backoff and neural modeling depends on corpus composition, lexical coverage and evaluation design.

**Keywords:** language modeling; smoothing; absolute discounting; neural networks; LSTM; data sparsity; vocabulary growth.

## 1. Introduction

A language model predicts words from preceding context. Plausible sequences remain unseen even in substantial collections. Unsmoothed n-grams assign them zero probability. Smoothing uses adjusted counts and shorter contexts; neural models share learned representations.

An unseen trigram may end in a frequent bigram that supports backoff. If the bigram is also unseen, neural representations may help. Out-of-vocabulary (OOV) targets are different: evaluation replaces their spelling with `<unk>`. Predicting this label does not recover the word.

Rebuilding vocabulary at each budget also changes the task. Unknown targets become separate words, mapped histories change, and output layers grow. Aggregate perplexity can rise while prediction improves on retained target occurrences.

We compare nine models across six corpora and five budgets, examining full-test rankings, budget trends and event-level differences. One split and ordered budget sequence per corpus measure conditional strengths, not a universal neural crossover or an isolated data-quantity effect.

## 2. Related Work

Katz [1] introduced backoff for sparse counts; Chen and Goodman [2] compared smoothing empirically. Our fixed absolute discount differs from Katz's estimator and Kneser-Ney [7]. We do not compare all strong smoothing alternatives.

Bengio et al. [3] jointly learned word representations and probabilities; LSTM [4] adds recurrent state. Both motivate unseen-combination generalization. Our LSTM also uses longer histories than the FFNNs, leaving architecture, context and capacity confounded.

Papusoi and Nisioi [15] compare count and neural BabyLM baselines, reporting tokenizer sensitivity and an evaluation artifact. Their grammaticality scores differ from held-out PPL. Here we examine vocabulary growth, mapped-label accuracy and additive event losses.

BabyLM [9, 10] studies limited-data pretraining, including 10-million-word and 100-million-word tracks. Its fourth report [16] extends beyond English. Our 53,269-2,980,263-word training collections use different tasks and scales, preventing direct transfer of rankings.

Nearest-neighbor language models [11] use retrieval for rare patterns and domain adaptation; Infini-gram [13] scales n-gram lookup with suffix arrays. Neither matches our fixed-order baseline. ByT5 [14] predicts bytes, avoiding one word-level unknown class, but its PPL uses different units.

Banko and Brill [5] study corpus size; Hoffmann et al. [12] jointly allocate model capacity and tokens for large transformers. They motivate considering compute, not inferring scaling laws here. Hwang et al. [17] find multilingual exposure-order effects; our ordering remains untested. NLTK [6] supplies access, while dataset documentation [8] motivates provenance and rights disclosure.

## 3. Data and Preprocessing

### 3.1 Corpora and Document Splits

Brown is the primary corpus; five others provide supplementary comparisons. The historical `brown_press_neural` directory contains all 15 genres. Split seed is 42, except WebText's 3999. Nominal document splits are 80/10/10 for Brown and Gutenberg, 70/20/10 elsewhere. Integer allocation gives Gutenberg 14/2/2 books and WebText 3/1/1 files.

**Table 1. Post-quarantine corpus sizes.** Each split cell gives documents / sentences / word tokens. Stored vocabulary includes four reserved symbols; test OOV excludes EOS.

| Corpus | Training D / S / W | Validation D / S / W | Test D / S / W | Full vocabulary | Test OOV |
|---|---:|---:|---:|---:|---:|
| Brown | 400 / 44,766 / 924,218 | 50 / 5,489 / 114,416 | 50 / 6,577 / 116,408 | 24,325 | 5.80% |
| Gutenberg | 14 / 49,356 / 951,248 | 2 / 8,605 / 201,507 | 2 / 38,181 / 978,972 | 75,817 | 9.28% |
| Inaugural | 42 / 3,877 / 99,657 | 12 / 1,111 / 30,361 | 6 / 406 / 10,934 | 12,136 | 5.73% |
| Reuters | 5,748 / 41,022 / 889,683 | 1,643 / 11,525 / 250,690 | 821 / 5,907 / 126,774 | 55,921 | 4.39% |
| ROCStories | 68,710 / 339,616 / 2,980,263 | 19,635 / 97,538 / 853,632 | 9,816 / 48,884 / 427,651 | 58,271 | 0.90% |
| WebText | 3 / 6,398 / 53,269 | 1 / 17,958 / 156,464 | 1 / 316 / 3,550 | 9,609 | 49.21% |

Brown's test covers 14 genres. Gutenberg tests on Austen's *Emma* and the King James Bible, and validates on Chesterton's novel *The Ball and the Cross* and Whitman's *Leaves of Grass*. Inaugural tests on Washington 1793, Monroe 1817, Taylor 1849, Buchanan 1857, Truman 1949 and Carter 1977. Reuters contains 8,212 retained news documents; its saved split does not establish topic or chronological separation. ROCStories contains 98,161 story documents from a pinned public mirror, repartitioned with identical-story hashes grouped together.

WebText trains on `grail.txt`, `pirates.txt` and `wine.txt`, validates on `overheard.txt`, and tests on `singles.txt`. Its validation text is almost three times the training text. One test file with 316 sentences provides a source-shift stress case, not representative web coverage. ROCStories likewise supplies narrative text but not story-level model state.

Saved audits report zero exact normalized cross-split sentence overlaps after quarantine. Removed occurrences total 5,801/624/17/1,518/3,192/74 in corpus order. Near screening requires ten words, five distinct five-token shingles and Jaccard similarity at least 0.9. Gutenberg retains two train-test pairs. Reuters retains six cross-split pairs: three train-test involving two test sentences, two train-validation and one validation-test. Brown, Inaugural and WebText have no above-threshold candidates; ROCStories was not near-scanned. These checks do not exclude shorter or semantic overlap.

### 3.2 Tokenization and Unknown Words

Model tokenization lowercases and splits on whitespace. Brown retains punctuation/POS forms such as `,/,`; attached punctuation elsewhere makes `lord`, `lord,` and `lord:` separate types. Cross-corpus PPL therefore reflects different token identities as well as language difficulty.

Each prefix builds its own training-only vocabulary. Brown retains frequency-two words; the other corpora retain singletons. Saved full-text checks find 19,602 sub-threshold Brown occurrences and none elsewhere, with no literal `<unk>` tokens. Brown supplies positive unknown-target supervision; the other five do not. Their UNK parameters still receive softmax updates, so they are not wholly untrained.

### 3.3 Training Budgets

Budgets take the first $\lceil pS\rceil$ training sentences for $p=0.2,0.4,0.6,0.8,1$. Sorted document concatenation makes them prefixes, not random samples or whole-document subsets. Brown adds genres, Inaugural later addresses, Gutenberg alphabetically ordered books, and WebText source files. Reuters and ROCStories follow source IDs without a verified chronological curriculum. Boundaries can fall inside documents.

The 20% prefixes contain 202,514/219,389/29,618/174,359/596,519/7,745 words. Inaugural thus starts with about 30% of full training words, WebText with 14.5%. Appendix C lists all 30 actual sizes. Sentence percentages are neither word nor compute fractions.

## 4. Methodology

### 4.1 Experimental Workflow

Frozen V3 excludes earlier whole-document pilots and Inaugural's separate 10/50/100% convergence study. Brown outputs came from Colab; Kubeflow recovery and imports retained the others. Import dates neither date training nor add fits.

The [preparer](../../src/prepare_mega_run.py) freezes membership and quarantines exact overlaps. [The full runner](../../src/run.py) selects a prefix before [controller initialization](../../src/lm/controller.py) builds its training-only `Vocab`. All same-budget models share this vocabulary. [Comparison code](../../src/model_comparison.py) creates six views from one count pass, scores splits and saves events; rare-file scoring stays separate. [Offline analysis](../../src/analyze_comparison_sweep.py) summarizes outputs without fitting. Figures 1-2 show these phases.

![Frozen six-corpus experimental pipeline](../../results/paper_methodology_20261107/figures/experimental_pipeline.png)

*Figure 1. Frozen workflow: training-only vocabulary, validation-selected neural checkpoints and separate test scoring.*

![Pipeline phase-to-code reference](../../results/paper_methodology_20261107/figures/phase_code_reference.png)

*Figure 2. Source paths for preparation, vocabulary, training, comparison and saved-output analysis. Comparison fits counts; offline analysis does not.*

### 4.2 Count-Based Models

The [n-gram estimators](../../src/lm/statistical/ngram.py) use one-token bigram and two-token trigram histories, initially BOS and BOS/BOS. For continuation count $c(h,w)$ and total $c(h)$,

$$
P_{\mathrm{MLE}}(w\mid h)=\frac{c(h,w)}{c(h)}.
$$

An unseen event or history receives zero probability. Add-one smoothing instead uses

$$
P_{+1}(w\mid h)=\frac{c(h,w)+1}{c(h)+|\mathcal V|},
$$

where $\mathcal V$ excludes PAD/BOS and includes UNK/EOS. Its size is stored vocabulary size minus two. Absolute-discount backoff uses $d=0.75$. For an observed history with unobserved continuations,

$$
P_{\mathrm{BO}}(w\mid h)=
\begin{cases}
\frac{c(h,w)-d}{c(h)}, & c(h,w)>0,\\
\alpha(h)P_{\mathrm{BO}}(w\mid h'), & c(h,w)=0,
\end{cases}
$$

Here $h'$ drops the oldest token. For observed continuation set $S_h$, the redistribution normalizer is

$$
\alpha(h)=\frac{d|S_h|/c(h)}{1-\sum_{u\in S_h}P_{\mathrm{BO}}(u\mid h')}.
$$

A wholly unseen history backs off directly to the shorter history; the unigram base is add-one smoothed. If all allowed continuations have been observed, the implementation returns MLE rather than removing mass with no destination. [Shared variant construction](../../src/model_comparison.py) reuses the counts for both orders and all three estimators. Kneser-Ney and discount tuning are not evaluated.

### 4.3 Feed-Forward Models

The [FFNN forward](../../src/lm/neural/ffnn/model.py) embeds the preceding one or two token IDs in 128 dimensions, concatenates them, and applies a 256-unit tanh layer, dropout 0.3 and a vocabulary projection. These histories match bigram/trigram lengths, except that context two begins with PAD/BOS rather than BOS/BOS. The module returns logits; training/evaluation masks forbidden targets before softmax.

For stored vocabulary size $V$, parameter counts are $385V+33{,}024$ and $385V+65{,}792$. The additional context adds 32,768 hidden-layer parameters, so this is not a capacity-matched context ablation.

### 4.4 Sentence-Prefix LSTM

The [LSTM forward](../../src/lm/neural/lstm/model.py) uses 128-dimensional embeddings, one recurrent layer with 256 hidden units, dropout 0.3 on embeddings and outputs, and a vocabulary projection. [Sentence batching](../../src/lm/data.py) keeps sentences separate; no hidden state carries across rows. Prediction is causal and can use the available sentence prefix, not a fixed four-word window.

The parameter count is $385V+395{,}264$. With Inaugural's $V=12{,}136$, totals are 4,705,384/4,738,152/5,067,624 for FFNN contexts one/two and LSTM. These include masked output rows. Figure 3 shows the module paths; Appendix A retains all three actual Torch-summary screenshots.

![FFNN and sentence-prefix LSTM architectures](../../results/paper_architecture_20261107/figures/neural_architectures.png)

*Figure 3. FFNN windows and sentence-prefix LSTM. Masking occurs outside module forwards; recurrence resets between sentences.*

## 5. Experimental Setup

### 5.1 Training and Checkpoint Selection

The [trainer](../../src/lm/neural/trainer.py) uses Adam at 0.001, batches of up to 32 sentences, gradient clipping at 1.0, maximum 20 epochs and patience three. A strictly lower token-mean validation NLL replaces the best checkpoint; test metrics do not select weights. Reserved IDs are PAD=0, UNK=1, BOS=2 and EOS=3. [Evaluation](../../src/lm/evaluation.py) and training exclude PAD/BOS predictions while retaining UNK/EOS.

Brown, Gutenberg, Inaugural and Reuters use seeds 13/42/2026. ROCStories and WebText retain seed 13 only. The matrix contains 210 neural fits and 180 deterministic statistical conditions from 30 shared count passes: 390 test conditions, not independent replicates. Hyperparameters are fixed across corpora.

### 5.2 Metrics and Diagnostic Populations

For $N$ word-plus-EOS events,

$$
\mathrm{NLL}=-\frac{1}{N}\sum_{i=1}^{N}\log p_i,
\qquad
\mathrm{PPL}=\exp(\mathrm{NLL}).
$$

A zero probability makes PPL infinite. Word accuracy/OOV exclude EOS; OOV correctness scores UNK. Neural summaries use arithmetic mean PPL and sample SD where replicated; other SDs remain undefined. Variation describes initialization and batch order, not population uncertainty. Training and scoring times are separate.

Event categories are assigned in order: OOV target; unseen bigram with retained target; unseen trigram with seen bigram; seen trigram. Membership uses same-budget mapped MLE sequences and includes EOS. Retained targets can still have OOV history tokens. Native accuracy histories are one/two-token FFNN windows or exact mapped LSTM prefixes; common short-history strata compare shared populations, not hidden-state familiarity.

The fixed cohort selects targets retained at 20%, including EOS: 109,631/843,843/10,035/121,301/465,770/1,436 events. It fixes occurrences, not support, mapped contexts, composition or capacity. WebText's cohort covers only 37.1% of test events.

Loss decomposition fixes bigram backoff for Brown/Gutenberg/WebText and trigram backoff for Inaugural/Reuters/ROCStories, selected retrospectively by full-test PPL. For category share $q_k$,

$$
\Delta_k=q_k\left(\overline{\mathrm{NLL}}_{\mathrm{LSTM},k}
-\mathrm{NLL}_{\mathrm{BO},k}\right),
\qquad
\Delta_{\mathrm{all}}=\sum_k\Delta_k.
$$

The bar averages seed NLL, not log mean PPL. Negative contributions favor LSTM. This additive comparison accounts for loss differences; it is not a deployment selection rule.

## 6. Results

### 6.1 Full-Budget Comparison

**Table 2. Full-budget test PPL.** Neural entries are mean +/- sample SD for four three-seed corpora; ROCStories/WebText are single runs. Bold marks the corpus minimum.

| Model | Brown | Gutenberg | Inaugural |
|---|---:|---:|---:|
| Bigram MLE | $\infty$ | $\infty$ | $\infty$ |
| Trigram MLE | $\infty$ | $\infty$ | $\infty$ |
| Bigram add-one | 1,589.2 | 11,772.1 | 2,958.7 |
| Trigram add-one | 9,641.1 | 52,180.6 | 8,130.0 |
| Bigram backoff | 260.0 | **914.5** | 344.3 |
| Trigram backoff | 260.2 | 940.8 | **342.0** |
| FFNN, context 1 | 263.2 +/- 0.4 | 1,174.0 +/- 36.6 | 448.8 +/- 3.9 |
| FFNN, context 2 | 246.6 +/- 0.9 | 1,171.2 +/- 41.6 | 422.8 +/- 8.3 |
| LSTM | **206.8 +/- 0.9** | 1,053.9 +/- 2.4 | 387.1 +/- 17.4 |

*Table 2 continued: added corpora.*

| Model | Reuters | ROCStories (one seed) | WebText (one seed) |
|---|---:|---:|---:|
| Bigram MLE | $\infty$ | $\infty$ | $\infty$ |
| Trigram MLE | $\infty$ | $\infty$ | $\infty$ |
| Bigram add-one | 5,638.2 | 1,211.0 | 8,077.2 |
| Trigram add-one | 23,001.0 | 6,476.6 | 9,380.5 |
| Bigram backoff | 357.4 | 134.4 | **7,810.6** |
| Trigram backoff | **291.2** | 110.4 | 7,945.4 |
| FFNN, context 1 | 502.7 +/- 6.3 | 140.3 | 12,211.9 |
| FFNN, context 2 | 412.8 +/- 5.0 | 106.8 | 13,513.4 |
| LSTM | 308.1 +/- 1.0 | **78.2** | 13,526.2 |

Backoff leads on four corpora and LSTM on two. LSTM is the best neural model throughout the first five corpora; WebText's best architecture changes with budget. The following observations connect each full-budget result to its data and event losses.

**Brown.** The 924,218 training words span multiple genres, with fiction entering larger prefixes. LSTM reaches 206.82 +/- 0.88 PPL against bigram backoff's 260.01, a 20.5% reduction. Its full-budget accuracy is 17.32%, versus trigram backoff's 15.38%. Frequency-two vocabulary supplies positive UNK examples, and the OOV loss penalty is small. LSTM improves on seen and unseen mapped contexts against the fixed bigram comparator. The gain is broad within this split, although ordered genre exposure prevents attributing it to quantity alone.

**Gutenberg.** Training contains 951,248 words, but two test books supply 978,972 words and 9.28% OOV. Bigram backoff leads at 914.48 PPL; LSTM reaches 1,053.95 +/- 2.43, 15.3% worse. The 75,817-entry vocabulary retains singletons but supplies no positive UNK targets. LSTM improves unseen-bigram prediction, yet unknown-target losses reverse its combined non-OOV gain. Literary and Biblical overlap also limits independent domain interpretation. Its full-budget fixed-cohort advantage is narrower than its complete-test deficit and does not establish a general training threshold.

**Inaugural.** The 42 training addresses contain 99,657 words. Six test addresses span 1793-1977, with 5.73% word OOV. Trigram backoff leads at 341.96 PPL against LSTM's 387.07 +/- 17.45, a 13.2% deficit. Repeated formal phrases support lower-order counts when longer contexts are new. LSTM gains on unseen bigrams, but unknown-target losses outweigh non-OOV improvements. Bigram add-one reaches 16.76% accuracy despite 2,958.7 PPL, showing that a favorable top prediction can coexist with poor probability allocation across the rest of the vocabulary.

**Reuters.** The retained news collection supplies 889,683 training words and 126,774 test words. Trigram backoff reaches 291.24 PPL; LSTM gives 308.08 +/- 0.95, only 5.8% worse. Financial phrases can favor within-sentence neural prediction when the final bigram is absent. Across the full test, however, OOV losses outweigh the non-OOV gain despite 4.39% word OOV. The neural deficit narrows with budget without crossing backoff. Neither source-ID ordering nor the seed-42 document split demonstrates topic-disjoint or chronological evaluation, and retained related-news pairs remain disclosed.

**ROCStories.** The largest training collection contains 2,980,263 words from short narratives, with 0.90% test-word OOV. The single LSTM run reaches 78.18 PPL versus trigram backoff's 110.35, a 29.2% improvement; FFNN context two also wins at 106.85. LSTM improves seen trigrams as well as unseen contexts. Its severe conditional UNK loss affects only a small event share, leaving a strong aggregate gain. This is sentence-level prediction, not cross-sentence narrative reasoning. One seed and three capped LSTM fits limit claims about variability and convergence.

**WebText.** Training contains 53,269 words from three files; the held-out personal-advertisement source contains 3,550 words and 49.21% OOV. Bigram backoff leads at 7,810.58 PPL. FFNN context one is the best neural model at 12,211.86, 56.4% worse; LSTM reaches 13,526.22. Unlike Gutenberg and Reuters, LSTM loses on both OOV and combined non-OOV events. Familiar short phrases still help backoff under source shift. The oversized validation source and single neural seed further restrict interpretation. High mapped-UNK accuracy does not indicate successful prediction of unfamiliar words.

### 6.2 Smoothing Ablation

All 60 MLE test conditions have infinite PPL. Full-budget bigram zero rates are 24.7/41.5/41.0% for Brown/Gutenberg/Inaugural; trigram rates rise to 63.3/79.6/76.5%. More context without smoothing worsens sparse-count failure.

Add-one removes zeros but corresponding-order backoff has roughly 6-55 times lower full-budget PPL on these three corpora. Gutenberg trigram add-one rises from 16,783.6 to 52,180.6 with budget. Its pseudocount denominator grows with support, diluting concentrated history evidence. Counts and domain composition change too, so support growth is not an isolated cause. Add-one nevertheless has lower OOV-only PPL than backoff on Gutenberg/Inaugural. On WebText its aggregate deficits are just 3.4% for bigrams and 18.1% for trigrams. Finite probability is necessary, but its useful allocation remains corpus-dependent.

### 6.3 Training-Budget Trends

**Table 3. Best neural / best backoff PPL by budget.** Values below one favor neural prediction. WebText uses FFNN context one / LSTM / LSTM / FFNN context two / FFNN context one; all other neural cells use LSTM. Backoff order may change.

| Corpus | 20% | 40% | 60% | 80% | 100% |
|---|---:|---:|---:|---:|---:|
| Brown | 0.859 | 0.833 | 0.808 | 0.800 | 0.795 |
| Gutenberg | 1.201 | 1.182 | 1.133 | 1.129 | 1.153 |
| Inaugural | 1.323 | 1.220 | 1.186 | 1.146 | 1.132 |
| Reuters | 1.298 | 1.181 | 1.118 | 1.108 | 1.058 |
| ROCStories (one seed) | 0.832 | 0.776 | 0.728 | 0.710 | 0.708 |
| WebText (one seed) | 0.738 | 1.670 | 1.405 | 1.266 | 1.564 |

Endpoint neural ratios improve in five corpora, but Gutenberg worsens from 80% to 100%. WebText favors neural prediction only at 20%; its changing winner is not an LSTM learning curve. Brown LSTM PPL rises from 164.5 to 206.8 while its backoff-relative advantage grows. Figures 4-5 place these trends alongside expanding vocabulary and falling OOV. Neither plot holds composition, support or capacity fixed.

![Test perplexity across all six corpora and five budgets](../../results/paper_submission_20261107/figures/ppl_vs_budget_six_corpora.png)

*Figure 4. Test PPL by corpus and sentence budget. Infinite MLE results are marked separately; neural SD is shown only for three-seed results.*

![Training vocabulary and test-word OOV across budgets](../../results/paper_submission_20261107/figures/vocabulary_oov_by_budget.png)

*Figure 5. Budget vocabulary, including reserved symbols, and validation/test-word OOV. Vocabulary is rebuilt at every prefix.*

### 6.4 Fixed-Cohort Results

**Table 4. PPL on the same 20%-retained target occurrences at two budgets.** Backoff is the better order for each cohort/budget, not necessarily Table 2's winner.

| Corpus | Backoff at 20% | LSTM at 20% | Backoff at 100% | LSTM at 100% | LSTM/backoff, 20% -> 100% |
|---|---:|---:|---:|---:|---:|
| Brown | 255.4 | 216.6 +/- 2.1 | 198.1 | 150.2 +/- 1.2 | 0.848 -> 0.758 |
| Gutenberg | 352.9 | 417.8 +/- 61.4 | 229.9 | 219.1 +/- 13.1 | 1.184 -> 0.953 |
| Inaugural | 223.6 | 219.9 +/- 10.1 | 169.0 | 159.4 +/- 1.5 | 0.983 -> 0.943 |
| Reuters | 261.6 | 258.1 +/- 14.4 | 146.4 | 122.3 +/- 4.2 | 0.986 -> 0.836 |
| ROCStories (one seed) | 124.3 | 92.5 | 87.6 | 56.8 | 0.744 -> 0.648 |
| WebText (one seed) | 246.4 | 256.1 | 228.1 | 277.3 | 1.039 -> 1.215 |

Backoff and LSTM improve monotonically on the first five fixed cohorts. Add-one generally worsens and MLE remains infinite; Inaugural FFNN context two reverses slightly at the final step. WebText LSTM worsens from 256.1 to 277.3. Fixed membership alone does not remove all adverse budget effects.

Gutenberg LSTM crosses below backoff at full budget. Individual seed PPLs of 226.8/226.4/204.0 all beat 229.9, although two margins are small and mean improvement 10.8 is below SD 13.1. This is a selected-cohort result, not a significance test or universal threshold. Reuters and ROCStories favor LSTM on their cohorts at every budget.

### 6.5 Convergence and Recorded Cost

Of 210 neural fits, 207 stop on patience; three ROCStories LSTM runs reach epoch 20 at 60/80/100%, with best epochs 19/19/20. Their stopping does not demonstrate convergence. LSTM best epochs are 6-8 for Brown/Reuters, 9-11 for Inaugural, 1-3 for Gutenberg, and 2 then 1 for WebText. Early validation selection does not prove undertraining.

Full-budget LSTM sizes are 9.76/29.58/5.07/21.92/22.83/4.09 million parameters in corpus order. Across architectures and budgets, sizes range from 0.84 to 29.58 million. Appendix B reports shared-count, neural-training and scoring times without multiplying the count pass sixfold. Heterogeneous environments and unequal epochs prevent hardware-normalized or equal-compute efficiency claims.

### 6.6 Accuracy and Unknown-Label Ties

Figure 6 separates complete-word accuracy from retained-word accuracy and the always-UNK baseline. Winner sets change in 15 of 30 corpus/budget pairs, including five tie-only changes; ten pairs have no common winner. PPL rankings remain separate.

![Full-budget word accuracy, retained-word accuracy and always-UNK baseline](../../results/paper_submission_20261107/figures/full_budget_accuracy_vs_unk.png)

*Figure 6. Full-budget accuracy with word-only denominators. The always-UNK baseline measures label collapse, not lexical recovery; single-run SD is undefined.*

WebText trigram add-one scores 1,470/3,550, or 41.4085%, overall. Only 28/1,803 retained words are correct (1.5530%); 1,442/1,747 OOV labels are correct (82.54%). Always predicting UNK reaches 49.2113%. Trigram backoff instead gets 92 retained words right (5.10%); LSTM gets one.

The [statistical prediction adapter](../../src/lm/cli.py) selects the lowest ID among maximum-count followers. On unseen histories, add-one is uniform and all six corpora select UNK, the lowest allowed ID. WebText's unseen-history UNK successes contribute 40.6197 percentage points. Keeping the denominator but abstaining on these ties gives 0.7887%; uniform random tie-breaking gives expected 0.7972%. These accuracy-only counterfactuals do not change probabilities or PPL. Brown also has 1,349 seen-history UNK successes: 1,139 unique maxima and 210 tied maxima. Not every UNK success comes from an unseen-history tie.

## 7. Error Analysis and Discussion

### 7.1 Sparsity and Aggregate Loss

LSTM reduces unseen-bigram retained-target PPL against the fixed backoff comparator by 37.2/43.0/46.8/67.3/82.3/11.5% in corpus order. This is generalization beyond observed mapped bigrams, not proof of rare-word mastery. Trigram backoff wins seen-trigram PPL on five corpora; ROCStories favors LSTM, 24.3 versus 26.9. Choosing different backoff orders within categories would create an oracle, not one model.

**Table 5. Category contributions to LSTM-minus-backoff NLL.** Entries are nats per total word-plus-EOS event; negative favors LSTM. Fixed comparators are bigram for Brown/Gutenberg/WebText, trigram elsewhere.

| Category contribution | Brown | Gutenberg | Inaugural | Reuters | ROCStories | WebText |
|---|---:|---:|---:|---:|---:|---:|
| Seen trigram | -0.06934 | +0.02797 | +0.13304 | +0.18257 | -0.06731 | +0.02968 |
| Unseen trigram, seen bigram | -0.04864 | +0.10623 | +0.03651 | -0.04131 | -0.14337 | +0.14881 |
| Unseen bigram, retained target | -0.11281 | -0.18319 | -0.22356 | -0.27837 | -0.19584 | -0.05357 |
| OOV target | +0.00192 | +0.19093 | +0.17724 | +0.19332 | +0.06182 | +0.42423 |
| **Total** | **-0.22887** | **+0.14194** | **+0.12323** | **+0.05622** | **-0.34469** | **+0.54915** |

Gutenberg's non-OOV gain of -0.04899 is outweighed by +0.19093 OOV loss; Inaugural's -0.05402 by +0.17724; Reuters' -0.13710 by +0.19332. Brown's OOV penalty is only +0.00192. ROCStories retains a -0.34469 total despite OOV PPL near 48 billion because the OOV event share is 0.8%. WebText loses +0.42423 on OOV and +0.12492 on non-OOV, so removing unknown targets would not reverse its ranking.

Figure 7 shows this accounting. Full-budget non-OOV populations differ from the 20%-retained cohorts, and neither replaces complete-test evaluation. The decomposition identifies where losses occur, not the effect of changing tokenization or UNK supervision. Category shares and conditional scores are in Appendix B.

![Category-weighted LSTM versus backoff loss differences](../../results/paper_submission_20261107/figures/full_budget_loss_decomposition.png)

*Figure 7. Additive full-budget NLL differences with fixed backoff comparators. Category contributions sum to total loss difference; negative favors LSTM.*

### 7.2 Fifteen Illustrative Cases

Cases are selected diagnostic examples from saved V3 exports, not random prevalence estimates. Values are natural-log target probabilities; neural values average seed log probabilities. Unless stated otherwise, the comparator is trigram backoff, as in the case exports, rather than Table 5's corpus-specific order. Less negative values assign more target probability.

1. **Brown: "vary according -> to".** The trigram is absent, but `according to` occurs 103 times. Backoff gives -0.05 against add-one's -10.10. Strong shorter-context evidence supports the target without spreading probability uniformly across the vocabulary.

2. **Inaugural: "important branch -> of".** The trigram is unseen and the final bigram count is eight. Backoff gives -0.10 versus add-one's -9.40. Even modest lower-order evidence can make a useful prediction when the longer sequence is new.

3. **Brown: "comparable -> basis".** The bigram is unseen; `basis` occurs 158 times. LSTM gives -1.67 versus backoff's -9.40. Neural prediction succeeds without direct bigram evidence, although the example does not separate representation sharing from the longer sentence prefix.

4. **Gutenberg: "lead among -> you.".** The final bigram is absent and `you.` occurs 179 times. LSTM gives -5.25 versus backoff's -10.30. The punctuation-bearing retained target benefits from neural context even though punctuation increases vocabulary fragmentation.

5. **Inaugural: "partiality has -> bestowed".** The target occurs four times and the bigram is unseen. LSTM improves from backoff's -11.28 to -7.24. The gain is real for this event, but the target remains low-probability rather than confidently recovered.

6. **Brown: "peanut and sesame -> oils".** At 20%, LSTM gives the UNK mapping -2.21. At 100%, `oils` is retained with count two and receives -20.03. These are different prediction labels, so their difference cannot be read as simple forgetting.

7. **Gutenberg: "my lord -> esau:".** The target is absent and maps to UNK. LSTM gives -22.96 versus backoff's -16.07. Domain-specific spelling and attached punctuation contribute to exclusion, but the probability being scored belongs to the shared unknown symbol.

8. **Inaugural: "be duly -> appreciated".** The target is absent from training. LSTM gives -19.25 against backoff's -11.66. An ordinary completion receives very little mapped-UNK probability in a model with no positive unknown-target training examples.

9. **Brown: "for god's -> sake".** The trigram count grows from zero to seven. LSTM improves from -13.57 to -0.17; full-budget backoff gives -0.11. Both benefit when directly relevant observations enter the ordered prefix.

10. **Inaugural: "mr. vice -> president,".** The trigram grows from zero to four occurrences. LSTM improves from -10.90 to -4.62, but backoff reaches -0.21. Improvement with added text does not imply a neural win on a newly observed phrase.

11. **Gutenberg: "o my -> lord,".** Trigram count grows from zero to one and target count from one to 134. LSTM improves from -11.87 to -3.67, near backoff's -3.69. Shared Biblical/literary phrasing limits claims of overlap-free domain generalization.

12. **Gutenberg: "frank churchill was one of the -> boasts".** This singleton has an unseen final bigram. LSTM/FFNN context two/backoff give -13.49/-13.91/-14.51. All struggle despite parameter sharing; add-one gives -11.30, so not every model falls below -13.

13. **Reuters: "in 1985's 4th -> qtr.".** The bigram is absent; `qtr.` occurs nine times. LSTM gives -5.90, FFNN context two -10.79 and backoff -15.93. Financial sentence context helps, without identifying longer-prefix information as the sole cause.

14. **ROCStories: "he had lost 5 -> pounds.".** The target occurs 106 times but the bigram is unseen. LSTM gives -0.31 versus FFNN's -3.45 and backoff's -12.01. This is within-sentence prediction; preceding story sentences do not supply recurrent state.

15. **WebText: "ala. lots -> of".** The trigram is unseen but `lots of` occurs 15 times. Backoff gives -0.05, LSTM -4.39 and add-one -9.17. A familiar shorter phrase explains a non-OOV backoff gain under source shift.

### 7.3 Isolated Rare-Sequence Probe

The separate probe weights zero-count held-out n-grams by test frequency and uses only their one/two-token histories. Full-budget FFNN context two / trigram backoff PPL is 587.0/710.0 for Brown, 2,577.3/2,320.3 for Gutenberg, 899.9/828.5 for Inaugural, 2,079.8/1,788.8 for Reuters, 1,076.5/1,398.6 for ROCStories and 15,658.8/9,122.4 for WebText. Different order lists contain different rows, and isolated LSTM probes remove its normal sentence prefix. These results do not replace full-sentence unseen-context analysis.

## 8. Limitations, Ethics and Reproducibility

One split and budget order confound source, genre and era with quantity. Two Gutenberg books and one WebText test file limit coverage. ROCStories and WebText lack seed replication. Cohorts do not control support or capacity; architectures use different histories. Tuned discount, Kneser-Ney, subword and pretrained baselines are absent.

On 7 November 2026, the group reported instructor approval for supplementary corpora and reporting disclosed retained-overlap flags. No written record, instructor name or original approval date was supplied. This disposition does not establish zero near overlap, independence or universal rubric compliance. Gutenberg/Reuters flags and the missing ROCStories near-scan remain scientific limitations.

Historical, news, narrative and web text can contain bias and identifiable people; WebText includes personal advertisements. Public availability is not blanket redistribution permission. The completed [dataset card](../submission/dataset_card.md) and [data-access document](../submission/data_access_instructions.md) record sources, preprocessing and rights. ROCStories' mirror declares CC BY 4.0 without independently establishing upstream rights. No new participant data was collected. The [contribution and AI disclosure](../submission/contributions_and_ai.md) records equal authorship contributions and Copilot assistance, not independent scientific validation.

Original recovery records identify Python 3.11.11, Torch 2.11.0+cu126 and an A100 80GB allocation, not verified per-condition hardware. Brown signatures record Torch 2.11.0+cu130; its complete environment is unavailable. CPU Torch 2.14.1/torchinfo 1.8.0 belongs to summary inspection, not training. The [reproducibility document](../submission/reproducibility.md) separates these runtimes.

There are 210 usable final checkpoints but only 20 count archives; ten Gutenberg/Inaugural archives are missing and a zero-byte resume checkpoint is excluded. Replay is incomplete. Saved-event audits reconcile 330 finite PPLs within relative error $3.8\times10^{-7}$ and 60 infinite results. Accuracy partitions reconcile all 390 conditions. Figure checks cover 270 summaries, 30 budgets and 210 convergence rows, verifying consistency rather than independent replication.

## 9. Conclusion

Smoothing removes the zero-probability failure of unsmoothed n-grams. Backoff leads on four complete test sets and LSTM on two. LSTM consistently improves unseen mapped bigrams with retained targets, but OOV penalties reverse non-OOV gains on Gutenberg, Inaugural and Reuters. WebText loses on both groups and exposes mapped-UNK accuracy inflation.

Fixed-cohort LSTM PPL improves in five corpora, not WebText. Vocabulary, composition and capacity changes prevent causal data-efficiency claims. Fixed support, alternative orders, stronger baselines and additional seeds would test generality. Model mixtures and new ablations remain unperformed.

## References

1. Katz, S. M. 1987. Estimation of probabilities from sparse data for the language model component of a speech recognizer. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 35(3):400-401.
2. Chen, S. F., and Goodman, J. 1996. [An Empirical Study of Smoothing Techniques for Language Modeling](https://aclanthology.org/P96-1041/). *Proceedings of ACL*, 310-318.
3. Bengio, Y., Ducharme, R., Vincent, P., and Jauvin, C. 2003. [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html). *Journal of Machine Learning Research*, 3:1137-1155.
4. Hochreiter, S., and Schmidhuber, J. 1997. Long short-term memory. *Neural Computation*, 9(8):1735-1780.
5. Banko, M., and Brill, E. 2001. [Scaling to Very Very Large Corpora for Natural Language Disambiguation](https://aclanthology.org/P01-1005/). *Proceedings of ACL*, 26-33.
6. Bird, S., Klein, E., and Loper, E. 2009. [Natural Language Processing with Python](https://www.nltk.org/book/). O'Reilly Media. The linked online edition has subsequent updates.
7. Kneser, R., and Ney, H. 1995. Improved backing-off for M-gram language modeling. *Proceedings of ICASSP*, 181-184.
8. Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daume III, H., and Crawford, K. 2021. Datasheets for datasets. *Communications of the ACM*, 64(12):86-92.
9. Warstadt, A., et al. 2023. [Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora](https://aclanthology.org/2023.conll-babylm.1/). *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning*, 1-34.
10. Hu, M. Y., et al. 2024. [Findings of the Second BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora](https://aclanthology.org/2024.conll-babylm.1/). *The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning*, 1-21.
11. Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., and Lewis, M. 2020. [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172). *International Conference on Learning Representations*. Preprint first posted in 2019.
12. Hoffmann, J., et al. 2022. [An empirical analysis of compute-optimal large language model training](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, 35.
13. Liu, J., Min, S., Zettlemoyer, L., Choi, Y., and Hajishirzi, H. 2024. [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377). *Conference on Language Modeling*.
14. Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., and Raffel, C. 2022. [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://aclanthology.org/2022.tacl-1.17/). *Transactions of the Association for Computational Linguistics*, 10:291-306.
15. Papusoi, R., and Nisioi, S. 2025. [A Comparison of Elementary Baselines for BabyLM](https://aclanthology.org/2025.babylm-main.16/). *Proceedings of the First BabyLM Workshop*, 218-225.
16. Choshen, L., et al. 2026. [Findings of the Fourth BabyLM Challenge & Workshop: Going Beyond English](https://aclanthology.org/2026.babylm-main.37/). *Proceedings of the Second BabyLM Workshop*, 479-492.
17. Hwang, J., Edman, L., and Fraser, A. 2026. [Language Exposure Matters: Distribution and Order in Multilingual BabyLM Pre-training](https://aclanthology.org/2026.babylm-main.69/). *Proceedings of the Second BabyLM Workshop*, 904-911.

## Appendix A. Neural Architecture Inspection

These saved screenshots inspect Inaugural 100%-budget, seed-13 checkpoints on CPU with Torch 2.14.1 and torchinfo 1.8.0. They are not training logs. Input `[32,10]` and output `[32,10,12136]` are dummy inspection shapes, not an LSTM context limit. Parameter totals agree across saved seeds. The original generator loaded weights and ran dummy forwards; it was not rerun here. Memory and mult-add estimates are not measured training time or GPU peaks.

![Actual saved FFNN context-1 Torch summary](../../results/inaugural_100pct_torch_summaries/ffnn_ctx1.png)

*Figure A1. Saved FFNN context-one Torch summary, with 4,705,384 trainable parameters.*

![Actual saved FFNN context-2 Torch summary](../../results/inaugural_100pct_torch_summaries/ffnn_ctx2.png)

*Figure A2. Saved FFNN context-two Torch summary, with 4,738,152 trainable parameters.*

![Actual saved sentence-prefix LSTM Torch summary](../../results/inaugural_100pct_torch_summaries/lstm.png)

*Figure A3. Saved LSTM Torch summary, with 5,067,624 trainable parameters. Ten tokens is an inspection length.*

## Appendix B. Recorded Runtime and Event Categories

### B.1 Runtime

Table B1 separates one shared count pass from LSTM training and test scoring. Neural entries average available seeds; ROCStories/WebText use one run. Figure B1 also shows FFNN times. Summed neural training minutes across all budgets are 101.4/127.2/9.2/86.2/333.7/2.6 in corpus order, covering 45 fits each in the first four corpora and 15 each in the last two. Training totals exclude comparison scoring. Recovery allocation records do not establish each condition's hardware, so these are recorded costs rather than normalized benchmarks.

**Table B1. Full-budget recorded seconds.** Shared fitting serves all six statistical variants; do not sum it six times.

| Corpus | Shared count fit | LSTM train | Bigram-backoff test | Trigram-backoff test | LSTM test |
|---|---:|---:|---:|---:|---:|
| Brown | 8.6 | 318.8 | 7.0 | 41.2 | 12.6 |
| Gutenberg | 6.3 | 187.7 | 34.2 | 152.6 | 64.8 |
| Inaugural | 0.6 | 22.1 | 0.4 | 1.2 | 0.6 |
| Reuters | 5.9 | 251.9 | 5.3 | 18.8 | 8.8 |
| ROCStories | 17.5 | 3,346.8 | 17.6 | 57.1 | 34.3 |
| WebText | 0.3 | 13.1 | 0.1 | 0.2 | 0.2 |

![Recorded full-budget fitting and test-scoring time](../../results/paper_submission_20261107/figures/full_budget_recorded_time.png)

*Figure B1. Recorded full-budget time by model and corpus. Shared statistical fitting is counted once; execution environments are not hardware-matched.*

### B.2 Event Shares and Conditional Perplexity

Figure B2 reports all four mutually exclusive category shares across the five budgets, including EOS. Table B2 pairs each fixed backoff comparator with LSTM at full budget. Conditional PPL describes a category, whereas its contribution to total loss also depends on category frequency. Arithmetic seed-mean PPL cannot be combined to recover aggregate PPL; Table 5 instead averages seed NLL contributions.

![Test-event category shares across five budgets](../../results/paper_submission_20261107/figures/test_event_category_shares.png)

*Figure B2. Category shares across all five budgets, with word-plus-EOS denominators. OOV takes priority over history categories.*

**Table B2. Category share; fixed backoff / LSTM PPL.** Backoff is bigram for Brown/Gutenberg/WebText and trigram elsewhere. Neural PPL averages available seeds.

| Category | Brown | Gutenberg | Inaugural |
|---|---|---|---|
| Seen trigram | 33.9%; 33.1 / 27.0 | 20.4%; 40.4 / 46.5 | 23.5%; 19.2 / 33.9 |
| Unseen trigram, seen bigram | 36.4%; 181.1 / 158.5 | 38.1%; 169.5 / 224.2 | 35.5%; 90.7 / 100.6 |
| Unseen bigram, retained target | 24.1%; 12,507.1 / 7,852.8 | 32.6%; 5,059.3 / 2,883.2 | 35.4%; 3,109.7 / 1,655.4 |
| OOV target | 5.5%; 39.2 / 40.7 | 8.9%; 2,943,011 / 28,656,791 | 5.5%; 259,996 / 7,786,237 |

*Table B2 continued.*

| Category | Reuters | ROCStories | WebText |
|---|---|---|---|
| Seen trigram | 39.3%; 17.6 / 28.1 | 64.3%; 26.9 / 24.3 | 3.5%; 216.8 / 507.2 |
| Unseen trigram, seen bigram | 31.6%; 130.9 / 115.0 | 23.6%; 224.7 / 122.4 | 7.6%; 16.2 / 114.5 |
| Unseen bigram, retained target | 24.9%; 14,338.1 / 4,694.5 | 11.3%; 31,998.6 / 5,657.5 | 43.7%; 2,100.0 / 1,857.9 |
| OOV target | 4.2%; 2.85e6 / 3.40e8 | 0.8%; 2.21e7 / 4.84e10 | 45.2%; 1.04e5 / 2.66e5 |

## Appendix C. Actual Training-Budget Sizes

Table C1 transcribes the 30 rows of [budget data sizes](../../results/paper_submission_20261107/tables/budget_data_sizes.csv). Validation/test populations stay fixed as in Table 1. Vocabulary includes PAD/UNK/BOS/EOS; OOV uses test words only. The label selects a ceiling-rounded sentence prefix, not the same fraction of words or documents.

**Table C1. Selected sentences, words, stored vocabulary and test-word OOV by budget.**

| Corpus | Budget | Sentences | Words | Vocabulary | Test OOV |
|---|---:|---:|---:|---:|---:|
| Brown | 20% | 8,954 | 202,514 | 9,980 | 11.47% |
| Brown | 40% | 17,907 | 403,856 | 15,333 | 8.32% |
| Brown | 60% | 26,860 | 625,759 | 19,322 | 7.10% |
| Brown | 80% | 35,813 | 784,017 | 22,113 | 6.25% |
| Brown | 100% | 44,766 | 924,218 | 24,325 | 5.80% |
| Gutenberg | 20% | 9,872 | 219,389 | 20,024 | 17.70% |
| Gutenberg | 40% | 19,743 | 401,986 | 34,750 | 13.21% |
| Gutenberg | 60% | 29,614 | 567,272 | 43,791 | 12.39% |
| Gutenberg | 80% | 39,485 | 786,736 | 61,655 | 10.59% |
| Gutenberg | 100% | 49,356 | 951,248 | 75,817 | 9.28% |
| Inaugural | 20% | 776 | 29,618 | 5,421 | 11.94% |
| Inaugural | 40% | 1,551 | 53,066 | 7,780 | 8.52% |
| Inaugural | 60% | 2,327 | 70,593 | 9,449 | 6.98% |
| Inaugural | 80% | 3,102 | 85,503 | 10,844 | 6.21% |
| Inaugural | 100% | 3,877 | 99,657 | 12,136 | 5.73% |
| Reuters | 20% | 8,205 | 174,359 | 20,215 | 8.98% |
| Reuters | 40% | 16,409 | 351,304 | 31,702 | 6.56% |
| Reuters | 60% | 24,614 | 530,847 | 41,534 | 5.51% |
| Reuters | 80% | 32,818 | 710,338 | 49,311 | 4.86% |
| Reuters | 100% | 41,022 | 889,683 | 55,921 | 4.39% |
| ROCStories | 20% | 67,924 | 596,519 | 28,661 | 2.52% |
| ROCStories | 40% | 135,847 | 1,193,538 | 39,292 | 1.64% |
| ROCStories | 60% | 203,770 | 1,789,846 | 46,957 | 1.26% |
| ROCStories | 80% | 271,693 | 2,384,755 | 53,127 | 1.05% |
| ROCStories | 100% | 339,616 | 2,980,263 | 58,271 | 0.90% |
| WebText | 20% | 1,280 | 7,745 | 2,107 | 68.45% |
| WebText | 40% | 2,560 | 18,355 | 4,580 | 59.38% |
| WebText | 60% | 3,839 | 31,616 | 7,079 | 53.01% |
| WebText | 80% | 5,119 | 43,013 | 8,631 | 50.87% |
| WebText | 100% | 6,398 | 53,269 | 9,609 | 49.21% |