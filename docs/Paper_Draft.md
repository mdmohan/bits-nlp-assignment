# **Smoothing, Sparsity and Vocabulary Growth: Comparing Backoff and Neural Language Models Across Training Budgets**

**Group:** 1

| Author | Student ID |
| :---- | :---- |
| Akshaya Kumar Joish | 2025cs05013 |
| Anuj Kumar Srivastava | 2025cs05040 |
| Murali Das Mohan | 2025cs05052 |

Working manuscript, version 0.4, 6 November 2026\. Expanded to six corpora with Reuters, ROCStories and WebText. Editable research draft, not a submission-ready paper. Saved metrics and test-event probabilities were reanalysed; no models were retrained or rescored for this update. ROCStories and WebText have one neural seed and remain exploratory. Related work includes publication records through October 2026\. Affiliations, contribution statements and submission formatting remain open.

## **Abstract**

Sparse observations make language modeling a choice between reliable counts and generalization beyond observed sequences. We compare six bigram/trigram estimators, two feed-forward models and a sentence-level LSTM across Brown, Gutenberg, Inaugural, Reuters, ROCStories and WebText. Five training budgets, three neural seeds for four corpora and one for two yield 390 test conditions. At full budget, LSTM beats the best backoff model by 20.5% on Brown and 29.2% on ROCStories; backoff leads on the other four. LSTM nevertheless reduces perplexity on unseen bigrams with retained targets by 11.5-82.3% across all six corpora. On Gutenberg, Inaugural and Reuters, OOV losses outweigh its gains on non-OOV events. ROCStories tolerates a severe conditional OOV penalty because only 0.8% of scored events are OOV. WebText is a counterexample to an OOV-only explanation: LSTM also loses on non-OOV events and worsens on its fixed target cohort between the smallest and largest budgets. WebText further exposes misleading mapped-label accuracy: 1,442 of trigram add-one’s 1,470 correct predictions are unknown symbols. These findings distinguish contextual generalization, lexical coverage and evaluation effects. Single-seed results, ordered budgets, changing prediction support, limited document coverage and heterogeneous hardware preclude a universal data-efficiency threshold.

**Keywords:** language modeling; smoothing; absolute discounting; neural networks; LSTM; data sparsity; vocabulary growth.

## **1\. Introduction**

A language model estimates the probability of a word from its preceding context. Even a substantial training corpus leaves many plausible word sequences unobserved. Maximum-likelihood n-gram estimation assigns such sequences zero probability, while smoothing redistributes probability mass to support unseen events. Neural language models instead share parameters across contexts and words, potentially assigning useful probabilities to combinations absent from training.

These approaches address related but distinct kinds of sparsity. An unseen trigram may have a well-supported final bigram, making backoff effective. An unseen bigram with a retained target may benefit from learned representations. A target outside the vocabulary is a different problem again: the evaluator scores an unknown-word symbol rather than recovery of the original word. Treating all three as a single category of “rare language” obscures what a model has learned.

Training-budget comparisons introduce another difficulty. If the vocabulary is rebuilt at each budget, more training text also changes the prediction problem. Words previously collapsed into an unknown-word class become separate targets, output layers grow, and mapped contexts change. A rising aggregate perplexity therefore need not mean that additional training observations damaged predictive ability.

We investigate three questions: (1) which models perform best on the full held-out test sets; (2) how their relative performance changes across observed training budgets; and (3) which categories of test events account for the differences. The experiment is a descriptive comparison on one document split and one ordered budget sequence per corpus. It does not identify a universal data threshold or isolate the causal effect of training size.

## **2\. Related Work**

Katz \[1\] established a backoff framework that uses lower-order distributions when higher-order evidence is unavailable, while Chen and Goodman \[2\] compared smoothing techniques empirically. These foundational citations identify our methods, not the current research frontier. Our fixed absolute-discount implementation differs from Katz’s original discount estimator and from Kneser-Ney smoothing \[7\]; its results cannot establish superiority over the strongest count-based baselines.

Bengio et al. \[3\] introduced jointly learned word representations and probability models, motivating generalization to unseen combinations of retained words. LSTM networks \[4\] supply recurrent state for sequential prediction. Here architecture and context length differ: the LSTM can use the sentence prefix, whereas the feed-forward models use one or two tokens. The experiment does not isolate representation sharing or longer memory as the sole source of improvement.

Recent work makes the comparison directly relevant to current research. Papusoi and Nisioi \[15\] compare frequency baselines, n-grams, LSTMs with multiple tokenizers, and GPT-BERT in the 2025 BabyLM setting. They report tokenization sensitivity and identify an evaluation artifact in the grammaticality-testing pipeline. Their outcomes are BLiMP-based rather than our held-out PPL, so numerical rankings are not directly comparable. This close precedent means our contribution is the particular event-level loss decomposition and vocabulary-growth analysis, not the introduction of a count-versus-neural comparison.

The BabyLM Challenge \[9\] and its second edition \[10\] organize sample-efficient pretraining under limited data budgets, including 10-million-word and 100-million-word text-only tracks in 2024\. The fourth challenge report \[16\], published in October 2026, extends the setting beyond English and associates stronger performance with factors including preprocessing, training epochs and recursive architectures. Our full training corpora range from approximately 53,000 to 2.98 million words and use a different evaluation protocol. These studies motivate examining data efficiency, but do not establish an optimal training budget or architecture for our corpora.

Modern work also challenges a strict separation between memorization and neural generalization. Khandelwal et al. \[11\] interpolate neural predictions with nearest-neighbor retrieval, improving rare-pattern prediction and enabling datastore-based domain adaptation. Liu et al. \[13\] scale unbounded n-gram modeling using suffix-array indexing and show that it can complement neural language models. Infini-gram is neither our fixed-order model nor a matched small-data experiment, but it establishes the continuing relevance of count-based evidence. Combining our backoff and neural models remains an untested extension.

Token representation is another active design choice. ByT5 \[14\] operates directly on bytes rather than a learned word or subword vocabulary and studies robustness alongside computational trade-offs. Such a representation avoids collapsing unseen surface words into one word-level unknown symbol, although it does not guarantee accurate prediction of those words. This motivates a future representation ablation, not a claim that ByT5 has solved the failure measured here. Byte-level and word-level PPL also use different prediction units and must not be directly compared.

Banko and Brill \[5\] established the importance of corpus scale for natural-language disambiguation. Hoffmann et al. \[12\] subsequently studied model-size and training-token allocation under a fixed compute budget for large transformers. This motivates treating data, capacity and compute jointly. Our ordered budgets change all three, so they do not estimate a scaling law or inherit a compute-optimal prescription from large-model experiments.

Exposure order is relevant beyond our setting. Hwang et al. \[17\] find that both the allocation and order of language exposure affect multilingual BabyLM performance, with mixed exposure outperforming staged exposure in their matched configurations. That multilingual result is not proof of our monolingual genre-order confound, but it reinforces the need to test alternative exposure schedules before attributing budget trends solely to data quantity.

NLTK \[6\] supplies corpus access, not uniform preprocessing or licensing. Following Gebru et al. \[8\], we treat source provenance, tokenization and documented limitations as part of the experimental evidence.

## **3\. Data and Preprocessing**

### **3.1 Corpora and Document Splits**

We use Brown, Gutenberg, Inaugural, Reuters, ROCStories and WebText. The directory called brown\_press\_neural contains the full Brown corpus, not only its press genres. Document splits use seed 42 except WebText, which uses 3999\. Reuters contributes 8,212 retained news documents; ROCStories contributes 98,161 story documents from the archived public mirror. WebText contains five retained source files, not thousands of independent documents. Table 1 reports post-quarantine sizes; split percentages are nominal document-allocation settings, not training budgets.

**Table 1\. Dataset and vocabulary characteristics.** OOV percentages use word tokens as the denominator. Vocabulary sizes are the stored vocabulary sizes, including reserved symbols.

| Property | Brown | Gutenberg | Inaugural |
| :---- | ----: | ----: | ----: |
| Source documents | 500 | 18 | 60 |
| Train / validation / test documents | 400 / 50 / 50 | 14 / 2 / 2 | 42 / 12 / 6 |
| Split proportions | 80 / 10 / 10 | 80 / 10 / 10 | 70 / 20 / 10 |
| Full training words | 924,218 | 951,248 | 99,657 |
| Test words | 116,408 | 978,972 | 10,934 |
| Test events, including EOS | 122,985 | 1,017,153 | 11,340 |
| Minimum retained frequency | 2 | 1 | 1 |
| Vocabulary, 20% \-\> 100% | 9,980 \-\> 24,325 | 20,024 \-\> 75,817 | 5,421 \-\> 12,136 |
| Test-word OOV, 20% \-\> 100% | 11.5% \-\> 5.8% | 17.7% \-\> 9.3% | 11.9% \-\> 5.7% |

*Table 1 continued: added corpora.*

| Property | Reuters | ROCStories | WebText |
| :---- | ----: | ----: | ----: |
| Retained source documents | 8,212 | 98,161 | 5 |
| Train / validation / test documents | 5,748 / 1,643 / 821 | 68,710 / 19,635 / 9,816 | 3 / 1 / 1 |
| Nominal split proportions | 70 / 20 / 10 | 70 / 20 / 10 | 70 / 20 / 10 |
| Full training words | 889,683 | 2,980,263 | 53,269 |
| Test words | 126,774 | 427,651 | 3,550 |
| Test events, including EOS | 132,681 | 476,535 | 3,866 |
| Minimum retained frequency | 1 | 1 | 1 |
| Vocabulary, 20% \-\> 100% | 20,215 \-\> 55,921 | 28,661 \-\> 58,271 | 2,107 \-\> 9,609 |
| Test-word OOV, 20% \-\> 100% | 9.0% \-\> 4.4% | 2.5% \-\> 0.9% | 68.5% \-\> 49.2% |
| Neural seeds | 13, 42, 2026 | 13 only | 13 only |

Brown’s test split covers 14 genres. Gutenberg’s test books are Austen’s *Emma* and the King James Bible; validation uses Chesterton’s *The Ball and the Cross* and Whitman’s *Leaves of Grass*. Its test text is slightly larger than its full training text, and two books do not provide broad independent domain coverage. Inaugural’s test addresses are Washington (1793), Monroe (1817), Taylor (1849), Buchanan (1857), Truman (1949) and Carter (1977). The reported scores characterize these particular held-out collections.

WebText trains on grail.txt, pirates.txt and wine.txt, validates on overheard.txt, and tests on singles.txt. Its validation text contains 156,464 words, nearly three times its training text; the test has only 316 sentences from one source. This is a narrow cross-source stress case, not a representative web-language benchmark. ROCStories models reset state at sentence boundaries: a narrative corpus does not turn this into a story-level reasoning experiment.

The saved leakage audits report zero exact lowercase/whitespace-normalized sentence overlaps between splits after quarantining 5,801 Brown, 624 Gutenberg and 17 Inaugural occurrences. Two Gutenberg near-duplicate pairs involving Biblical quotations were retained under an audit-only policy. Exact-match removal and a thresholded near-duplicate audit do not establish that all semantic overlap or source dependence has been eliminated.

The corresponding added-corpus quarantine totals are 1,518 Reuters, 3,192 ROCStories and 74 WebText occurrences, with zero reported exact overlaps. Reuters retains six flagged near-duplicate pairs under the accepted audit-only policy; its original audit still labels them for review. ROCStories’ near-duplicate scan was not run, an accepted and recorded limitation, whereas WebText’s scan found no candidates above its threshold. These audit scopes are not equivalent.

### **3.2 Tokenization and Unknown Words**

All six corpora use lowercase whitespace tokenization, but their inputs differ. Brown retains punctuation tags such as ,/, and ./.. Attached punctuation in the other corpora makes forms such as lord, lord, and lord: distinct types. Vocabulary growth and OOV rates reflect preprocessing as well as lexical diversity. Cross-corpus perplexity levels should not be interpreted on a common scale.

Each budget builds a vocabulary from its own training text. Words below the minimum frequency map to \<unk\>. Brown uses min\_freq=2, while Gutenberg and Inaugural use 1, a deviation from the earlier uniform-threshold research plan. A fresh count of the full-budget text finds 19,602 Brown word occurrences below the threshold and zero in the other two corpora; none contains a literal \<unk\> token. Thus Brown supplies positive unknown-target examples, while Gutenberg and Inaugural do not. In the latter models, the unknown output is still affected by softmax training, but it never receives positive target supervision. Calling its parameters wholly “untrained” would be imprecise.

Reuters, ROCStories and WebText also use min\_freq=1: their full-budget text contains no sub-threshold or literal \<unk\> occurrences. All five threshold-one corpora therefore lack positive unknown-target examples. ROCStories shows why that setting alone cannot determine the aggregate winner: its unknown-target problem is severe conditionally but affects relatively few events.

### **3.3 Training Budgets**

Budgets of 20%, 40%, 60%, 80% and 100% are prefixes of the training sentences, not random samples or whole-document subsets. Documents are concatenated in sorted-ID order. Brown therefore adds genres in a fixed order; fiction enters only in the larger budgets. Inaugural adds progressively later addresses. Gutenberg adds books alphabetically by source identifier. A prefix can end inside a document.

The actual 20% word totals are 202,514, 219,389, 29,618, 174,359, 596,519 and 7,745 for Brown, Gutenberg, Inaugural, Reuters, ROCStories and WebText. Inaugural’s 20% sentence budget contains about 30% of its full training words; WebText’s contains only 14.5%. Reuters and ROCStories follow sorted source IDs, not verified chronological or randomized curriculum order. WebText adds material in source-file order. Budget labels are not exact word or compute fractions, and added data changes composition, vocabulary and model size simultaneously.

## **4\. Models and Evaluation**

### **4.1 Count-Based Models**

For history ![][image1], target ![][image2], counts ![][image3] and history total ![][image4], maximum-likelihood estimation is

![][image5]

An unobserved event or history receives zero probability. Add-one smoothing uses

![][image6]

where ![][image7] is the allowed prediction support. \<unk\> and EOS are included; padding and BOS are excluded. The support size in this formula is therefore not the complete stored vocabulary size in Table 1\.

Absolute-discount backoff uses ![][image8]. For an observed history with at least one unobserved continuation,

![][image9]

where ![][image10] drops the oldest history token. If ![][image11] is the set of observed continuations,

![][image12]

An entirely unseen history backs off directly. The unigram base is add-one smoothed. If a history has observed every allowed continuation, the implementation uses MLE rather than discounting mass with nowhere to redistribute it. Bigram and trigram MLE, add-one and backoff variants share one counting pass per corpus-budget pair. Kneser-Ney is not evaluated.

### **4.2 Neural Models and Training**

The feed-forward models concatenate embeddings of the previous one or two tokens, apply a 256-unit tanh layer, and predict through a softmax output. Embeddings have dimension 128 and hidden-layer dropout is 0.3. The LSTM uses 128-dimensional embeddings, one 256-unit recurrent layer and dropout of 0.3 on embeddings and recurrent outputs. Its state is confined to each sentence, not carried across documents or sentences.

Training uses Adam with learning rate ![][image13], batch size 32, gradient clipping at 1.0, and a maximum of 20 epochs. Early stopping uses patience 3 on validation negative log-likelihood, retaining the best checkpoint. Brown, Gutenberg, Inaugural and Reuters use seeds 13, 42 and 2026; ROCStories and WebText use only seed 13\. There are 210 neural fits and 180 deterministic statistical model-budget evaluations, constructed from 30 shared counting passes: 390 test conditions in total. No per-corpus hyperparameter search or capacity matching is performed.

### **4.3 Metrics and Seed Summaries**

For ![][image14] scored target events, negative log-likelihood per event and perplexity are

![][image15]

PPL includes word targets and one EOS per sentence. A single zero-probability event makes PPL infinite. Top-1 accuracy and the ordinary OOV rate use word tokens only. Accuracy for an OOV word concerns its mapped \<unk\> label, not recovery of the original spelling. Fit/train and test-scoring times are reported separately.

Neural tables report arithmetic mean PPL and sample standard deviation where three seeds exist. ROCStories and WebText report single-run values, without an invented zero standard deviation. Deterministic n-grams have no seed standard deviation. Seed deviations measure initialization and batch-order sensitivity conditional on the fixed split, not uncertainty over documents or domains. We do not report significance tests or treat dependent token events as independent experimental replicates.

### **4.4 Event Categories, Fixed Cohort and Loss Decomposition**

Every test event is assigned, in priority order, to: (1) OOV target; (2) unseen bigram with a retained target; (3) unseen trigram but seen bigram; or (4) seen trigram. Seen/unseen status comes from same-budget MLE scores on vocabulary-mapped sequences. Categories include EOS events. A retained target does not imply that every context token is retained, and a seen mapped sequence need not be a seen raw surface sequence.

The fixed cohort retains non-OOV events at the 20% budget, including EOS: 109,631 Brown (89.1%), 843,843 Gutenberg (83.0%), 10,035 Inaugural (88.5%), 121,301 Reuters (91.4%), 465,770 ROCStories (97.7%) and 1,436 WebText (37.1%). The same target occurrences are evaluated at every budget. This controls cohort membership, not output support, context mapping, training composition or capacity. The particularly small WebText cohort must not stand in for its full test population.

For the full-budget comparison, we fix one backoff model per corpus: bigram for Brown, Gutenberg and WebText; trigram for Inaugural, Reuters and ROCStories, based on lowest aggregate test PPL. This is a retrospective descriptive comparator, not a validation-selected deployment rule. For category ![][image16] with event share ![][image17], the contribution to LSTM-minus-backoff loss is

![][image18]

The bar averages NLL across seeds. Negative values favor the LSTM. Unlike PPL, NLL is additive, allowing exact accounting of opposing gains and losses. We average per-seed losses directly, not the logarithm of arithmetic mean PPL.

## **5\. Results**

### **5.1 Full-Budget Comparison**

**Table 2\. Test perplexity at 100% training budget.** Neural values are mean \+/- sample standard deviation where three seeds exist; ROCStories and WebText are single-seed. Lower is better; bold identifies the lowest value within a corpus.

| Model | Brown | Gutenberg | Inaugural |
| :---- | ----: | ----: | ----: |
| Bigram MLE | ![][image19] | ![][image20] | ![][image21] |
| Trigram MLE | ![][image22] | ![][image23] | ![][image24] |
| Bigram add-one | 1,589.2 | 11,772.1 | 2,958.7 |
| Trigram add-one | 9,641.1 | 52,180.6 | 8,130.0 |
| Bigram backoff | 260.0 | **914.5** | 344.3 |
| Trigram backoff | 260.2 | 940.8 | **342.0** |
| FFNN, context 1 | 263.2 \+/- 0.4 | 1,174.0 \+/- 36.6 | 448.8 \+/- 3.9 |
| FFNN, context 2 | 246.6 \+/- 0.9 | 1,171.2 \+/- 41.6 | 422.8 \+/- 8.3 |
| LSTM | **206.8 \+/- 0.9** | 1,053.9 \+/- 2.4 | 387.1 \+/- 17.4 |

*Table 2 continued: added corpora.*

| Model | Reuters | ROCStories (one seed) | WebText (one seed) |
| :---- | ----: | ----: | ----: |
| Bigram MLE | ![][image25] | ![][image26] | ![][image27] |
| Trigram MLE | ![][image28] | ![][image29] | ![][image30] |
| Bigram add-one | 5,638.2 | 1,211.0 | 8,077.2 |
| Trigram add-one | 23,001.0 | 6,476.6 | 9,380.5 |
| Bigram backoff | 357.4 | 134.4 | **7,810.6** |
| Trigram backoff | **291.2** | 110.4 | 7,945.4 |
| FFNN, context 1 | 502.7 \+/- 6.3 | 140.3 | 12,211.9 |
| FFNN, context 2 | 412.8 \+/- 5.0 | 106.8 | 13,513.4 |
| LSTM | 308.1 \+/- 1.0 | **78.2** | 13,526.2 |

The LSTM improves on Brown’s best backoff model by 20.5% and ROCStories’ by 29.2%. It is worse by 15.2%, 13.2%, 5.8% and 73.2% on Gutenberg, Inaugural, Reuters and WebText. LSTM is the strongest neural model at every budget in the first five corpora, but not WebText: its full-budget FFNN context 1 is the best neural model and still trails backoff by 56.4%. ROCStories’ two-token FFNN also beats trigram backoff (106.8 versus 110.4); the added neural success is not limited to LSTM. These are observed rankings, with no seed-robustness claim for the two single-seed corpora.

Accuracy does not always agree with PPL. Brown’s LSTM has the highest full-budget accuracy, 17.32%, compared with 15.38% for trigram backoff. On Inaugural, bigram add-one achieves 16.76%, above trigram backoff’s 16.20% and LSTM’s 15.25%, despite much worse PPL. Smoothing changes the probabilities assigned beyond the highest-ranked token; accuracy alone would conceal serious calibration and coverage problems.

WebText is a sharper warning. Trigram add-one reports 41.41% mapped-word accuracy, but 1,442 of its 1,470 correct predictions are \<unk\> targets; only 28 of 1,803 retained words are correct (1.55%). Its 1,747 OOV words constitute 49.21% of test words. Trigram backoff has only 2.59% mapped accuracy but 5.10% accuracy on retained words. LSTM gets one retained word correct (0.06% of retained words). This is an evaluation-label effect and a substantial model failure, not evidence that add-one recovers unfamiliar lexical items.

### **5.2 Smoothing Ablation**

Both MLE orders have infinite test PPL at every budget in all six corpora: 60 infinite test conditions. At full budget, bigram zero-probability rates are 24.7%, 41.5% and 41.0% on Brown, Gutenberg and Inaugural; corresponding trigram rates are 63.3%, 79.6% and 76.5%. Increasing context order without smoothing intensifies sparse-count failure.

Add-one removes zeros but is much worse than backoff on the original three corpora. At full budget, corresponding-order backoff has approximately 6-55 times lower PPL across those six order-by-corpus comparisons. Gutenberg trigram add-one rises from 16,783.6 to 52,180.6 as the budget grows. Its denominator allocates a pseudocount to every allowed target, producing strong dilution when the vocabulary is large relative to a history’s observed count. Vocabulary growth is a concrete explanatory factor, although these runs do not isolate it from changing counts and domains.

Add-one is not uniformly worse on every subset. At full budget it gives lower OOV-only PPL than backoff on Gutenberg and Inaugural. Its failure is the overall allocation of probability, not an inability to assign mass to unknown targets. Backoff should not be called the best OOV method without specifying the comparator and corpus.

Nor is the aggregate add-one disadvantage equally large everywhere. On WebText, bigram add-one is only 3.4% worse than bigram backoff, and trigram add-one is 18.1% worse than trigram backoff. Both beat the full-budget neural models. The original three-corpus 6-55-fold gap does not generalize to all six corpora.

### **5.3 Aggregate Training-Budget Trends**

**Table 3\. Lowest neural PPL divided by lowest backoff PPL at each budget.** LSTM supplies all cells except WebText. For WebText, the neural models at 20/40/60/80/100% are FFNN context 1, LSTM, LSTM, FFNN context 2 and FFNN context 1\. Backoff order may differ between cells. Values below one favor the neural model.

| Corpus | 20% | 40% | 60% | 80% | 100% |
| :---- | ----: | ----: | ----: | ----: | ----: |
| Brown | 0.859 | 0.833 | 0.808 | 0.800 | 0.795 |
| Gutenberg | 1.201 | 1.182 | 1.133 | 1.129 | 1.153 |
| Inaugural | 1.323 | 1.220 | 1.186 | 1.146 | 1.132 |
| Reuters | 1.298 | 1.181 | 1.118 | 1.108 | 1.058 |
| ROCStories (one seed) | 0.832 | 0.776 | 0.728 | 0.710 | 0.708 |
| WebText (one seed) | 0.738 | 1.670 | 1.405 | 1.266 | 1.564 |

Relative neural performance improves from the smallest to largest budget in five corpora, but not WebText. Gutenberg’s ratio also worsens between 80% and 100%. Reuters narrows its LSTM deficit from 29.8% to 5.8% without crossing backoff; ROCStories favors LSTM throughout. WebText instead changes from a neural advantage at 20% to a backoff advantage thereafter, and the best neural architecture changes. Its curve is a changing-model envelope, not an LSTM learning curve. Neither direction establishes a causal threshold.

Brown’s absolute LSTM PPL rises from 164.5 to 206.8 even as its advantage over backoff grows. This cannot be read as a clean deterioration from adding data: both the vocabulary and the identities of mapped targets change. The fixed-cohort results provide a more restricted comparison.

![Aggregate neural-to-backoff perplexity ratios across budgets][image31]

Aggregate neural-to-backoff perplexity ratios across budgets

*Figure 1\. Descriptive aggregate PPL ratios across observed budgets. The curves do not hold vocabulary, domain composition or model capacity constant.*

### **5.4 Fixed-Cohort Results**

**Table 4\. PPL on the same target occurrences at 20% and 100%.** Backoff is the better order on the indicated cohort and budget, not necessarily the aggregate winner from Table 2\.

| Corpus | Backoff at 20% | LSTM at 20% | Backoff at 100% | LSTM at 100% | LSTM/backoff, 20% \-\> 100% |
| :---- | ----: | ----: | ----: | ----: | ----: |
| Brown | 255.4 | 216.6 \+/- 2.1 | 198.1 | 150.2 \+/- 1.2 | 0.848 \-\> 0.758 |
| Gutenberg | 352.9 | 417.8 \+/- 61.4 | 229.9 | 219.1 \+/- 13.1 | 1.184 \-\> 0.953 |
| Inaugural | 223.6 | 219.9 \+/- 10.1 | 169.0 | 159.4 \+/- 1.5 | 0.983 \-\> 0.943 |
| Reuters | 261.6 | 258.1 \+/- 14.4 | 146.4 | 122.3 \+/- 4.2 | 0.986 \-\> 0.836 |
| ROCStories (one seed) | 124.3 | 92.5 | 87.6 | 56.8 | 0.744 \-\> 0.648 |
| WebText (one seed) | 246.4 | 256.1 | 228.1 | 277.3 | 1.039 \-\> 1.215 |

Both backoff orders and LSTM improve monotonically on the fixed cohorts of Brown, Gutenberg, Inaugural, Reuters and ROCStories. Inaugural’s two-token FFNN has a small final-budget reversal, from 191.5 to 193.6. Improvement does not extend to all model families: add-one generally worsens, and MLE remains infinite. For example, Gutenberg cohort trigram add-one rises from 15,967.1 to 47,663.7 despite fixed target membership.

WebText breaks the broader improvement pattern. Its LSTM cohort PPL rises from 256.1 to 277.3, while its better backoff order improves overall from 246.4 to 228.1 but worsens slightly at the final step. Fixing target membership therefore does not remove all adverse budget effects. Domain composition, context mapping, vocabulary and stopping decisions remain confounded. Reuters and ROCStories favor LSTM on their fixed cohorts at all five budgets; the small Reuters 20% mean advantage is not a significance result.

Gutenberg changes from a backoff advantage at 80% to an LSTM mean advantage at 100%. The three full-budget LSTM cohort PPLs are 226.8, 226.4 and 204.0 for seeds 13, 42 and 2026; all beat bigram backoff’s 229.9. However, two margins are small and the mean difference of 10.8 is less than the sample standard deviation of 13.1. This is a crossover on one selected cohort and split, not a statistically established universal threshold.

### **5.5 Convergence, Variability and Cost**

Of 210 neural fits, 207 stopped on patience and three reached the epoch cap. The capped runs are ROCStories LSTM at 60%, 80% and 100%, with best epochs 19, 19 and 20\. Its full-budget result is budget-limited, not demonstrated convergence. Reuters LSTM best epochs range from 6 to 8\. WebText LSTM selects epoch 2 at 20% and epoch 1 thereafter. Gutenberg’s FFNN best epoch is 1 throughout, with LSTM at 1-3; Brown LSTM selects 6-8 and Inaugural 9-11. Early selection is consistent with validation-domain sensitivity but does not prove later checkpoints would improve independent test performance.

Full-budget LSTM parameter counts are approximately 9.76M, 29.58M, 5.07M, 21.92M, 22.83M and 4.09M for Brown, Gutenberg, Inaugural, Reuters, ROCStories and WebText. Parameters grow with vocabulary, so budget comparisons also change model capacity. Across the six-corpus sweep, neural models range from approximately 0.84M to 29.58M parameters.

**Table 5\. Full-budget runtime in seconds.** N-gram fitting is a shared count pass for all six variants; it must not be counted six times. LSTM times are seed means, or single-run values on ROCStories/WebText.

| Corpus | Shared n-gram fit | LSTM train | Bigram-backoff test scoring | Trigram-backoff test scoring | LSTM test scoring |
| :---- | ----: | ----: | ----: | ----: | ----: |
| Brown | 8.6 | 318.8 | 7.0 | 41.2 | 12.6 |
| Gutenberg | 6.3 | 187.7 | 34.2 | 152.6 | 64.8 |
| Inaugural | 0.6 | 22.1 | 0.4 | 1.2 | 0.6 |
| Reuters | 5.9 | 251.9 | 5.3 | 18.8 | 8.8 |
| ROCStories (one seed) | 17.5 | 3,346.8 | 17.6 | 57.1 | 34.3 |
| WebText (one seed) | 0.3 | 13.1 | 0.1 | 0.2 | 0.2 |

Brown’s LSTM requires roughly 37 times the shared count-fit time for its 20.5% PPL improvement. ROCStories requires about 191 times the count-fit time for its 29.2% improvement; it also uses more epochs and a different environment. Summed neural training minutes are 101.4 Brown, 127.2 Gutenberg, 9.2 Inaugural, 86.2 Reuters, 333.7 ROCStories and 2.6 WebText. These totals cover 45 fits each for the first four and 15 each for the last two, excluding comparison scoring. Brown ran on Colab; the remaining corpora include Kubeflow training or recovery. Times are descriptive, not hardware-normalized efficiency estimates, and must not be interpreted as equal-compute comparisons.

## **6\. Error Analysis and Discussion**

### **6.1 What Kind of Sparsity Favors Each Model?**

**Table 6\. Full-budget category PPL: fixed backoff comparator / LSTM.** Backoff is bigram for Brown, Gutenberg and WebText; trigram for Inaugural, Reuters and ROCStories. Shares include EOS. LSTM values are seed means except the single-seed ROCStories and WebText results.

| Category | Brown: share; BO / LSTM | Gutenberg: share; BO / LSTM | Inaugural: share; BO / LSTM |
| :---- | :---- | :---- | :---- |
| Seen trigram | 33.9%; 33.1 / 27.0 | 20.4%; 40.4 / 46.5 | 23.5%; 19.2 / 33.9 |
| Unseen trigram, seen bigram | 36.4%; 181.1 / 158.5 | 38.1%; 169.5 / 224.2 | 35.5%; 90.7 / 100.6 |
| Unseen bigram, retained target | 24.1%; 12,507.1 / 7,852.8 | 32.6%; 5,059.3 / 2,883.2 | 35.4%; 3,109.7 / 1,655.4 |
| OOV target | 5.5%; 39.2 / 40.7 | 8.9%; 2,943,011 / 28,656,791 | 5.5%; 259,996 / 7,786,237 |

*Table 6 continued: added corpora.*

| Category | Reuters: share; BO / LSTM | ROCStories: share; BO / LSTM | WebText: share; BO / LSTM |
| :---- | :---- | :---- | :---- |
| Seen trigram | 39.3%; 17.6 / 28.1 | 64.3%; 26.9 / 24.3 | 3.5%; 216.8 / 507.2 |
| Unseen trigram, seen bigram | 31.6%; 130.9 / 115.0 | 23.6%; 224.7 / 122.4 | 7.6%; 16.2 / 114.5 |
| Unseen bigram, retained target | 24.9%; 14,338.1 / 4,694.5 | 11.3%; 31,998.6 / 5,657.5 | 43.7%; 2,100.0 / 1,857.9 |
| OOV target | 4.2%; 2.85e6 / 3.40e8 | 0.8%; 2.21e7 / 4.84e10 | 45.2%; 1.04e5 / 2.66e5 |

The consistent neural advantage is on unseen bigrams with retained targets: LSTM reduces PPL by 37.2%, 43.0%, 46.8%, 67.3%, 82.3% and 11.5% on Brown, Gutenberg, Inaugural, Reuters, ROCStories and WebText. It also beats the other backoff order on this category in each corpus. This supports generalization beyond observed mapped bigrams, not rare-word mastery or entirely known histories. The effect’s magnitude varies substantially.

Higher-order counts remain useful where they exist. Trigram backoff has seen-trigram PPL 24.7, 27.6 and 19.2 on the original corpora, beating LSTM’s 27.0, 46.5 and 33.9; it also wins this category on Reuters and WebText. ROCStories is the exception: LSTM beats trigram backoff even on seen trigrams (24.3 versus 26.9). Thus “counts always win on memorized contexts” is not supported. Choosing the better backoff order separately for each category creates an oracle lower envelope, not a single model, and cannot reconstruct aggregate loss.

### **6.2 Why Aggregate Rankings Reverse**

**Table 7\. Contributions to LSTM-minus-backoff NLL, in nats per total test event.** Each entry is category share times the seed-mean conditional loss difference. Negative values favor LSTM. Comparators are fixed as in Table 6; rounding can affect the final digit of sums.

| Category contribution | Brown | Gutenberg | Inaugural | Reuters | ROCStories | WebText |
| :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| Seen trigram | \-0.06934 | \+0.02797 | \+0.13304 | \+0.18257 | \-0.06731 | \+0.02968 |
| Unseen trigram, seen bigram | \-0.04864 | \+0.10623 | \+0.03651 | \-0.04131 | \-0.14337 | \+0.14881 |
| Unseen bigram, retained target | \-0.11281 | \-0.18319 | \-0.22356 | \-0.27837 | \-0.19584 | \-0.05357 |
| OOV target | \+0.00192 | \+0.19093 | \+0.17724 | \+0.19332 | \+0.06182 | \+0.42423 |
| **Total** | **\-0.22887** | **\+0.14194** | **\+0.12323** | **\+0.05622** | **\-0.34469** | **\+0.54915** |

On Gutenberg, non-OOV events collectively favor LSTM by 0.04899 nats per total test event, but OOV adds 0.19093, yielding a net disadvantage of 0.14194. Inaugural shows the same structure: a non-OOV gain of 0.05402 is exceeded by an OOV penalty of 0.17724. Brown’s OOV penalty is only 0.00192 and does not erase its other gains.

Conditioning on full-budget non-OOV events, including EOS, gives backoff/LSTM mean PPLs of 290.2/227.3 on Brown, 414.2/392.9 on Gutenberg and 232.1/219.2 on Inaugural. These subsets differ from the 20%-vocabulary fixed cohorts in Table 4\. They reveal where losses occur, but discarding unknown targets is not a valid replacement for full-test evaluation.

The decomposition establishes an arithmetic explanation for the observed aggregate reversal. It does not establish how much a threshold change, synthetic unknown-word training or a new tokenizer would improve the trained models. Those interventions would alter vocabularies and training distributions and require fresh controlled experiments. Add-one’s much better OOV-only scores also show that a large unknown-word probability can be purchased at the expense of the rest of the distribution.

Reuters reproduces the reversal: non-OOV contributions total \-0.13710, outweighed by \+0.19332 from OOV, giving \+0.05622 overall. ROCStories instead gains \-0.40652 on non-OOV events and loses \+0.06182 on OOV, retaining \-0.34469 overall despite conditional OOV PPL of roughly 48 billion. The OOV event share, not just its conditional PPL, determines its aggregate influence.

WebText is different. Its \+0.42423 OOV contribution explains much of the \+0.54915 disadvantage, but non-OOV events also contribute \+0.12492 against LSTM. Gains on unseen bigrams do not offset losses on seen bigrams and trigrams. Removing OOV events therefore would not reverse its LSTM/backoff ranking. This is a descriptive counterexample to the OOV-only explanation, not an isolated causal test of domain shift.

### **6.3 Fifteen Illustrative Cases**

These examples come from the frozen full-budget test-event exports. They were selected by diagnostic criteria rather than random sampling, so they illustrate mechanisms and counterexamples, not their prevalence. All values below are natural-log target probabilities; neural values average log probabilities across seeds. Except where stated otherwise, the count comparator is **trigram backoff** for consistency with the case exports, not the corpus-specific comparator in Tables 6-7. A larger, less negative value means greater assigned target probability.

1. **Brown: “vary according \-\> to”.** The trigram is absent, but according to occurs 103 times in the full training text. Trigram backoff assigns log probability \-0.05, compared with \-10.10 for trigram add-one: a well-supported shorter context prevents indiscriminate smoothing from dominating the prediction.  
2. **Inaugural: “important branch \-\> of”.** The trigram count is zero while the final bigram count is eight. Backoff gives \-0.10 and add-one \-9.40, illustrating that even a modest number of lower-order observations can support a useful prediction when the longer context is new.  
3. **Brown: “comparable \-\> basis”.** The bigram is unseen, although basis appears 158 times. LSTM assigns \-1.67 versus backoff’s \-9.40, showing a strong successful prediction without direct bigram evidence; this case alone does not identify whether the useful information came from local similarity or the longer prefix.  
4. **Gutenberg: “lead among \-\> you.”.** The bigram is absent and the punctuation-bearing target occurs 179 times. LSTM gives \-5.25 versus backoff’s \-10.30, demonstrating a neural advantage on a retained target even under the corpus’s punctuation-sensitive representation.  
5. **Inaugural: “partiality has \-\> bestowed”.** The target occurs four times and the bigram is unseen. LSTM assigns \-7.24 versus backoff’s \-11.28: the neural model improves the probability, but \-7.24 still represents a difficult, low-probability event rather than confident lexical recovery.  
6. **Brown: “peanut and sesame \-\> oils”.** At 20%, the target is outside the vocabulary and the LSTM scores its \<unk\> mapping at \-2.21. At 100%, oils is retained with count two and receives \-20.03; comparing these values as the same lexical task would mistake a target-support change for simple forgetting.  
7. **Gutenberg: “my lord \-\> esau:”.** The target has zero training count and maps to \<unk\>; LSTM assigns \-22.96 versus backoff’s \-16.07. The example combines a domain-specific surface form with attached punctuation, but the scored target is the shared unknown symbol, not the name itself.  
8. **Inaugural: “be duly \-\> appreciated”.** The target is absent from training and LSTM gives \-19.25, versus backoff’s \-11.66. With no positive \<unk\> training targets, the model’s very low unknown-symbol probability penalizes an otherwise ordinary sentence completion.  
9. **Brown: “for god’s \-\> sake”.** The trigram count grows from zero to seven, and LSTM log probability improves from \-13.57 to \-0.17. Backoff also performs strongly at full budget (-0.11); the example shows both methods benefiting when directly relevant sequences enter the ordered training prefix.  
10. **Inaugural: “mr. vice \-\> president,”.** The trigram grows from zero to four occurrences. LSTM improves from \-10.90 to \-4.62, yet full-budget backoff reaches \-0.21, reminding us that improvement with added text does not imply neural superiority on a newly memorized phrase.  
11. **Gutenberg: “o my \-\> lord,”.** The trigram grows from zero to one occurrence and the target count from one to 134\. LSTM improves from \-11.87 to \-3.67, close to backoff’s \-3.69; the shared Biblical and literary phrasing cautions against treating apparent cross-domain generalization as necessarily independent of lexical overlap.  
12. **Gutenberg: “frank churchill was one of the \-\> boasts”.** The full-budget target is a singleton with an unseen final bigram. LSTM (-13.49), FFNN context 2 (-13.91) and trigram backoff (-14.51) all give very low probabilities; neural parameter sharing does not solve every rare-target prediction. Add-one also struggles (-11.30), but it is inaccurate to say every model is below \-13.  
13. **Reuters: “in 1985’s 4th \-\> qtr.”.** The final bigram is unseen, while qtr. has nine training occurrences. LSTM gives \-5.90 versus trigram backoff’s \-15.93 and FFNN context 2’s \-10.79. Financial context supports a strong neural advantage despite the retained punctuation-bearing target; the example does not isolate the contribution of the longer prefix.  
14. **ROCStories: “he had lost 5 \-\> pounds.”.** The target occurs 106 times but the final bigram is absent. LSTM gives \-0.31, FFNN context 2 \-3.45 and trigram backoff \-12.01. This illustrates useful within-sentence contextual prediction, not reasoning across the preceding story sentences, because the LSTM state resets at each sentence.  
15. **WebText: “ala. lots \-\> of”.** The trigram is unseen but lots of occurs 15 times. Trigram backoff gives \-0.05, compared with LSTM’s \-4.39 and trigram add-one’s \-9.17. A familiar shorter phrase remains useful under source shift, illustrating a non-OOV loss that cannot be explained by unknown-word supervision alone.

### **6.4 Isolated Rare-Sequence Probe**

The separate rare-list evaluation scores zero-count held-out n-grams using only their own one- or two-token histories and weights them by test frequency. On the full-budget trigram lists, FFNN context 2 achieves 587.0 versus backoff’s 710.0 on Brown. Backoff wins on Gutenberg (2,320.3 versus 2,577.3) and Inaugural (828.5 versus 899.9).

The added corpora split similarly: trigram backoff beats FFNN context 2 on Reuters (1,788.8 versus 2,079.8) and WebText (9,122.4 versus 15,658.8), while the FFNN wins on ROCStories (1,076.5 versus 1,398.6). These probe rankings must remain separate from full-sentence results.

This probe is secondary. Its bigram and trigram lists contain different rows, and the LSTM loses its usual sentence prefix. A poor LSTM rare-list score cannot be interpreted as contradicting its full-sentence unseen-bigram advantage. The two evaluations ask different questions.

## **7\. Limitations, Ethics and Reproducibility**

The experiment varies neural initialization and batch order only in four corpora, not document splits or budget orders. ROCStories and WebText have one neural seed each and do not satisfy a three-seed replication requirement. Genre, source, author and era shift alongside training size. Gutenberg has two test books; WebText has one test source. Many token observations do not substitute for independent document groups. No broad population-level significance or universal training-data threshold is claimed.

Cross-corpus PPL levels are not directly comparable because token representations and unknown-word thresholds differ. The fixed cohort holds target occurrences constant but not output support, capacity or mapped context. Brown’s rising aggregate PPL is consistent with lexical-support changes; attributing the entire rise exclusively to that mechanism would require a fixed-support control. A uniform min\_freq=2 rerun would be informative but would be a new experiment, not a correction that can be inferred from these scores.

Models use fixed hyperparameters, different effective contexts and unmatched parameter counts. No tuned discount, Kneser-Ney baseline, subword vocabulary or pretrained language model is included. Comparator choices and category analyses are descriptive and partly retrospective. The case studies are deliberately selected extremes or transitions; they must not substitute for the exhaustive category accounting. Runtime measurements reflect heterogeneous execution environments.

The corpora contain historical language, news, crowdsourced narratives and web material with social biases and potentially identifiable people. WebText includes personal advertisements; public availability does not remove privacy or licensing concerns. This collection is not balanced modern-language coverage. A dataset card must identify source versions, licenses and redistribution conditions for each corpus. ROCStories was obtained from an archived public mirror; the mirror’s license label must not be treated as independent verification of upstream redistribution rights. No new participant data was collected by this experiment.

The frozen V3 configuration, split manifests, budget texts, training histories, scoring events and artifact manifests are the primary evidence. The 5 November audit checked the original 225 test conditions. On 6 November, all six run-level summaries were regenerated and the 165 added test conditions were reconstructed from their saved probabilities. Together the audits cover 390 conditions: 330 finite PPLs agree within relative error below ![][image32], and 60 reproduce infinite MLE PPL. Original event tables were reused with their prior audit; new category/cohort/case tables and exact per-seed decompositions were generated. All nine full-budget WebText word-accuracy metrics were independently reconstructed by known/OOV target group. This checks saved-output consistency, not independent checkpoint inference, a new leakage scan or a fresh hash audit of every artifact. No model was fitted for this update.

## **8\. Conclusion**

Under the observed protocol, LSTM gives the best full-test PPL on Brown and ROCStories, while absolute-discount backoff leads on Gutenberg, Inaugural, Reuters and WebText. Unsmoothed n-grams fail through zero probabilities. Add-one often allocates probability poorly, although its disadvantage is small on WebText relative to the other corpora. Neither the full-test ranking nor the best neural architecture is universal.

The consistent descriptive LSTM advantage is on unseen mapped bigrams with retained targets across all six corpora. Backoff remains strong on observed contexts, but ROCStories shows that LSTM can also win on seen trigrams. OOV penalties outweigh non-OOV gains on Gutenberg, Inaugural and Reuters, but not ROCStories. WebText loses on both OOV and non-OOV groups and exposes the danger of interpreting mapped unknown-label accuracy as lexical recovery.

LSTM fixed-cohort PPL improves from the smallest to largest budget in five corpora but worsens on WebText. Vocabulary growth and ordered exposure prevent a causal data-efficiency claim. Saved probabilities support further work on ranking reversals, validation-selected model mixtures and loss concentration without retraining the base models. Confirmatory experiments would require additional ROCStories/WebText seeds, controlled unknown-word handling, fixed support, multiple budget orders and stronger count baselines. The evidence supports conditional strengths, not a universal neural crossover threshold.

## **References**

1\.        Katz, S. M. 1987\. Estimation of probabilities from sparse data for the language model component of a speech recognizer. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 35(3):400-401.

2\.        Chen, S. F., and Goodman, J. 1996\. [*An Empirical Study of Smoothing Techniques for Language Modeling*](https://aclanthology.org/P96-1041/). *Proceedings of ACL*, 310-318.

3\.        Bengio, Y., Ducharme, R., Vincent, P., and Jauvin, C. 2003\. [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html). *Journal of Machine Learning Research*, 3:1137-1155.

4\.        Hochreiter, S., and Schmidhuber, J. 1997\. Long short-term memory. *Neural Computation*, 9(8):1735-1780.

5\.        Banko, M., and Brill, E. 2001\. [*Scaling to Very Very Large Corpora for Natural Language Disambiguation*](https://aclanthology.org/P01-1005/). *Proceedings of ACL*, 26-33.

6\.        Bird, S., Klein, E., and Loper, E. 2009\. [*Natural Language Processing with Python*](https://www.nltk.org/book/). O’Reilly Media. The linked online edition has subsequent updates.

7\.        Kneser, R., and Ney, H. 1995\. Improved backing-off for M-gram language modeling. *Proceedings of ICASSP*, 181-184.

8\.        Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daume III, H., and Crawford, K. 2021\. Datasheets for datasets. *Communications of the ACM*, 64(12):86-92.

9\.        Warstadt, A., et al. 2023\. [*Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora*](https://aclanthology.org/2023.conll-babylm.1/). *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning*, 1-34.

10\.  Hu, M. Y., et al. 2024\. [*Findings of the Second BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora*](https://aclanthology.org/2024.conll-babylm.1/). *The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning*, 1-21.

11\.  Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., and Lewis, M. 2020\. [*Generalization through Memorization: Nearest Neighbor Language Models*](https://arxiv.org/abs/1911.00172). *International Conference on Learning Representations*. Preprint first posted in 2019\.

12\.  Hoffmann, J., et al. 2022\. [*An empirical analysis of compute-optimal large language model training*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, 35\.

13\.  Liu, J., Min, S., Zettlemoyer, L., Choi, Y., and Hajishirzi, H. 2024\. [*Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens*](https://arxiv.org/abs/2401.17377). *Conference on Language Modeling*.

14\.  Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., and Raffel, C. 2022\. [*ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models*](https://aclanthology.org/2022.tacl-1.17/). *Transactions of the Association for Computational Linguistics*, 10:291-306.

15\.  Papusoi, R., and Nisioi, S. 2025\. [*A Comparison of Elementary Baselines for BabyLM*](https://aclanthology.org/2025.babylm-main.16/). *Proceedings of the First BabyLM Workshop*, 218-225.

16\.  Choshen, L., et al. 2026\. [*Findings of the Fourth BabyLM Challenge & Workshop: Going Beyond English*](https://aclanthology.org/2026.babylm-main.37/). *Proceedings of the Second BabyLM Workshop*, 479-492.

17\.  Hwang, J., Edman, L., and Fraser, A. 2026\. [*Language Exposure Matters: Distribution and Order in Multilingual BabyLM Pre-training*](https://aclanthology.org/2026.babylm-main.69/). *Proceedings of the Second BabyLM Workshop*, 904-911.

## **Working Notes: Reanalysis and Submission Tasks**

This section is an editorial work log, not part of the final paper.

### **A. Reanalysis Record**

The *analysis\_and\_paper\_plan.md* now carries the six-corpus report and ranked research recommendations. The 5 November three-corpus evidence remains under its original output directory. Its nine summary files matched their predecessors byte-for-byte and nine category/cohort/case CSVs matched numerically. The 6 November update writes separately to results/paper\_v3\_six\_corpora\_20261106, retaining the old event audit and reconstructing Reuters (75 conditions), ROCStories (45) and WebText (45). New per-condition populations are 132,681, 476,535 and 3,866 events respectively. Per-seed additive loss checks passed.

Current six-corpus evidence:

·      *../../results/paper\_v3\_six\_corpora\_20261106/reanalysis\_checks.json*: combined audit with explicit old/new verification provenance.

·      *../../results/paper\_v3\_six\_corpora\_20261106/tables/main\_results\_100pct.md*: all six corpora, correct seed labels and runtimes.

·      *../../results/paper\_v3\_six\_corpora\_20261106/tables/full\_budget\_loss\_decomposition.csv*: both fixed backoff comparators and per-seed losses.

·      *../../results/paper\_v3\_six\_corpora\_20261106/tables/fixed\_cohort\_per\_seed.csv*: per-seed cohort results.

·      *../../results/paper\_v3\_six\_corpora\_20261106/tables/webtext\_accuracy\_decomposition.csv*: retained-word versus mapped-OOV correctness.

Historical three-corpus evidence:

Evidence files:

·      *../../results/paper\_v3\_recheck\_20261105/reanalysis\_checks.json*: condition counts, maximum finite-PPL errors and consistency checks.

·      *../../results/paper\_v3\_recheck\_20261105/tables/main\_results\_100pct.md*: full-budget results and runtimes.

·      *../../results/paper\_v3\_recheck\_20261105/tables/full\_budget\_loss\_decomposition.csv*: per-seed NLLs, event counts, EOS counts and weighted contributions against both backoff orders.

·      *../../results/paper\_v3\_recheck\_20261105/tables/fixed\_cohort\_per\_seed.csv*: exact per-seed cohort scores at every budget.

·      *../../results/paper\_v3\_recheck\_20261105/tables/event\_analysis\_brown.md*, *../../results/paper\_v3\_recheck\_20261105/tables/event\_analysis\_gutenberg.md*, and *../../results/paper\_v3\_recheck\_20261105/tables/event\_analysis\_inaugural.md*: all 36 selected cases and category/cohort tables.

Use the **manuscript’s** interpretation of those generated reports: their boilerplate claim that the cohort “isolates added data” is too strong. Only event membership is held constant. Their category rows also retain both backoff orders; the fixed-comparator convention in Tables 6-7 is deliberate.

The original 5 November runtime check reread training text/configs to confirm 19,602/0/0 sub-threshold occurrences, examined 135 stop reasons, and checked split IDs and leakage-audit totals. The 6 November update verified all 210 stop reasons and the added training texts’ token counts and absence of literal \<unk\>. Neither update reran leakage detection. Results come from retained V3 artifacts, not earlier pilots.

### **B. Corrections to the Original Plan**

| Original shorthand | Working-draft treatment |
| :---- | :---- |
| Backoff handles OOV best | Better than LSTM in the reported full-budget comparisons, but add-one has lower OOV-only PPL on Gutenberg/Inaugural. |
| One “best backoff” in category tables | The original rows selected different orders. Use a fixed model for aggregate decomposition and label category-wise minima separately. |
| Every model improves on the fixed cohort | LSTM improves monotonically in five corpora but worsens overall on WebText. Inaugural FFNN context 2 also has a reversal. Add-one generally worsens; MLE stays infinite. |
| Neural advantage grows everywhere | Endpoint improvement holds in five corpora, not WebText; Gutenberg also has a final-step reversal. |
| Gutenberg early stopping proves under-training | Early best epochs are observed; validation shift is a plausible explanation requiring an ablation. |
| \<unk\> is untrained | No positive training targets in all five threshold-one corpora; output parameters still receive softmax updates. |
| Gutenberg validation is poetry and essays | The named files are Whitman’s *Leaves of Grass* and Chesterton’s novel *The Ball and the Cross*. |
| Every model scores “boasts” below \-13 | LSTM, FFNN context 2 and trigram backoff do; trigram add-one is \-11.30. |
| Brown’s aggregate rise is purely a support artifact | Cohort evidence supports a support-growth explanation but cannot isolate its causal contribution. |
| All texts are public domain; no personal information | Verify corpus-specific rights and acknowledge identifiable people in historical/literary text. |

### **C. Reproduction Commands**

Run from the repository root in an environment containing the existing analysis dependencies. These commands read saved results and generate new analysis outputs; they do not train models.

python src/analyze\_comparison\_sweep.py \--out results/paper\_v3\_six\_corpora\_20261106  
 python src/analyze\_comparison\_sweep.py \--skip-summary \--events \--out results/paper\_v3\_six\_corpora\_20261106

The extra loss-decomposition and per-seed cohort exports are new to this review, not outputs of the existing CLI. The following recipe regenerates them using its event reader; it streams each archive once. The audit JSON records the checks performed during this review and is not rewritten by this recipe.

import math  
 from pathlib import Path  
 import sys

 import numpy as np  
 import pandas as pd

 sys.path.insert(0, "src")  
 import analyze\_comparison\_sweep as analysis

 output \= Path("results/paper\_v3\_six\_corpora\_20261106/tables")  
 output.mkdir(parents=True, exist\_ok=True)  
 metrics \= analysis.read\_metrics(analysis.DEFAULT\_RUNS, list(analysis.RUN\_DIRS))  
 summary \= analysis.summarize\_source(metrics, "test")  
 loss\_rows, cohort\_rows \= \[\], \[\]  
 for corpus, folder in analysis.RUN\_DIRS.items():  
 	archive \= analysis.DEFAULT\_RUNS / folder / "metrics/scoring\_events.jsonl.gz"  
 	events \= analysis.TestEvents(archive)  
     events.verify(metrics, corpus)  
 	full \= summary\[(summary.corpus \== corpus) & (summary.percent \== 100)\]  
 	best \= full\[full.model.isin(analysis.BACKOFF)\].sort\_values("ppl\_mean").iloc\[0\].model  
     categories \= events.categories(100)  
 	cohort \= \~events.flag(20, "bigram\_mle", analysis.OOV)  
 	for percent in analysis.BUDGETS:  
     	for model in analysis.CURVE\_MODELS:  
             for key, values in zip(events.keys(percent, model), events.seed\_logps(percent, model)):  
                 cohort\_rows.append(dict(corpus=corpus, percent=percent, model=model,  
                 	seed=key\[2\], events=int(cohort.sum()), nll=float(-values\[cohort\].mean()),  
                 	ppl=analysis.perplexity(values\[cohort\])))  
 	masks \= \[(category, categories \== category) for category in analysis.CATEGORIES\]  
 	masks \+= \[("non\_oov", categories \!= "oov\_target"), ("all", np.ones(len(categories), dtype=bool))\]  
 	for backoff in analysis.BACKOFF:  
         baseline \= events.logp(100, backoff)  
     	for key, neural in zip(events.keys(100, "lstm"), events.seed\_logps(100, "lstm")):  
             contributions \= \[\]  
             for category, mask in masks:  
                 base\_nll, neural\_nll \= float(-baseline\[mask\].mean()), float(-neural\[mask\].mean())  
                 contribution \= float(mask.mean()) \* (neural\_nll \- base\_nll)  
                 if category in analysis.CATEGORIES:  
                     contributions.append(contribution)  
                 loss\_rows.append(dict(corpus=corpus, percent=100, backoff=backoff,  
                 	is\_full\_budget\_best\_backoff=backoff \== best, seed=key\[2\], category=category,  
                 	events=int(mask.sum()), eos\_events=int((mask & events.eos).sum()),  
                 	share=float(mask.mean()), backoff\_nll=base\_nll, lstm\_nll=neural\_nll,  
                 	backoff\_ppl=math.exp(base\_nll), lstm\_ppl=math.exp(neural\_nll),  
                 	weighted\_nll\_delta=contribution))  
             assert math.isclose(sum(contributions), float((baseline \- neural).mean()), abs\_tol=1e-10)  
 	del events  
 pd.DataFrame(loss\_rows).to\_csv(output / "full\_budget\_loss\_decomposition.csv", index=False)  
 pd.DataFrame(cohort\_rows).to\_csv(output / "fixed\_cohort\_per\_seed.csv", index=False)

The separate WebText accuracy export is not produced by the CLI above. To reproduce it, stream webtext-nn/metrics/scoring\_events.jsonl.gz, select source \== "test", percent \== 100 and eos \== false, then group by model and OOV status and sum the Boolean correct field. All nine models must have 3,550 word events: 1,803 retained and 1,747 OOV. Divide total correct by 3,550 to match saved accuracy and retained correct by 1,803 for retained-word accuracy. The exported columns preserve each count and denominator; no model inference is needed.

### **D. Before Submission**

☒        Recompute summary metrics and event-level PPL without fitting models.

☒        Add an exact additive loss decomposition using fixed backoff comparators.

☒        Extend all paper sections to six corpora and fifteen individual case interpretations.

☒        Distinguish seed variation, cohort selection and causal uncertainty.

☒        Reconstruct 165 newly added test conditions and WebText’s mapped-accuracy decomposition.

☐         Add independent neural seeds for ROCStories and WebText before presenting them as replicated results; do not silently convert this task into new training.

☒        Add author names, student IDs and group identifier (Group 1).

☐         Add affiliations and member contributions.

☐         Confirm the assignment’s exact dataset options, word limit, citation style and required template against the original rubric. The plan alone is not the rubric.

☐         Complete the proposal, dataset card, data-access instructions and submission README.

☐         Verify dataset licenses and redistribution terms; document exact source versions.

☐         Recover exact training hardware and dependency versions from original environment records where possible; leave unavailable facts explicitly unknown.

☒        Add recent literature with primary-source metadata and abstract checks, including directly related 2025 work and October 2026 publications.

☐         Read the selected recent papers in full before finalizing detailed literature comparisons; standardize author lists, name diacritics and citation order to the required style.

☐         Review the bibliography and add a corpus-specific Brown citation if required. The Chen-Goodman entry here deliberately uses the verified 1996 conference record.

☐         Include derived evidence and figure files when sharing this document; local result files are ignored by Git and relative links will otherwise break.

☐         Add a reviewed AI-use declaration. For this drafting session, GitHub Copilot assisted with reanalysis scripting, consistency checks, interpretation and manuscript prose; no new model training was performed. The team must verify the draft and describe other AI use separately.

☐         Decide whether controlled follow-up experiments are needed. Uniform unknown-word handling, fixed support, alternative budget orders and validation-domain ablations are proposals, not completed results.

☐         Remove these editorial notes and apply publication formatting only when the working draft is approved.

### **E. Literature Refresh (5 November 2026\)**

The bibliography now contains 17 entries, including nine newly added papers from 2020-2026; ten entries in total are from 2020 onward. Original method citations are retained so that newer application papers do not replace attribution for backoff, neural probabilistic models or LSTM. This was a targeted source search, not a systematic literature review. Publication records and abstracts were checked on ACL Anthology, the NeurIPS proceedings site and author-submitted arXiv records; full-text review remains an explicit submission task.

| Priority | References | Purpose in this paper |
| :---- | :---- | :---- |
| Closest direct comparison | \[15\], Papusoi and Nisioi (2025) | N-gram/LSTM/transformer baselines and tokenizer sensitivity; different evaluation target, so do not import its rankings. |
| Modern n-gram relevance | \[13\], Liu et al. (2024) | Large-scale n-grams can complement neural predictions; not evidence for the same effect at our scale. |
| Unknown-word alternatives | \[14\], Xue et al. (2022) | Byte-level modeling avoids a word-vocabulary bottleneck; no direct comparison of byte and word PPL. |
| Current sample-efficiency context | \[9\], \[10\], \[16\], BabyLM (2023, 2024, 2026\) | Limited-data training remains active; budgets, languages and evaluation differ from this experiment. |
| Exposure-order caveat | \[17\], Hwang et al. (2026) | Evidence that order and allocation matter in another constrained-data setting, not a causal explanation of our curves. |
| Memory and neural complementarity | \[11\], Khandelwal et al. (2020) | Retrieval improves rare-pattern prediction; a future hybrid here still needs testing. |
| Capacity/data/compute confounding | \[12\], Hoffmann et al. (2022) | Joint resource allocation matters; large-transformer scaling prescriptions do not transfer automatically. |

Reference \[12\] uses the verified NeurIPS title, *An empirical analysis of compute-optimal large language model training*, rather than its differently titled preprint, *Training Compute-Optimal Large Language Models*. Publication years for kNN-LM, ByT5 and Infini-gram refer to their conferences/journal, not the date of the first or latest arXiv version. The 2026 references have October 2026 publication records, preceding this working draft.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAXCAYAAADHhFVIAAAAYklEQVR4XmNgYGAoZ8ADhpmkKRD/A2IhNDmwpDEQX0GXAAGQ5AIgvgHEXkB8C4g9kCWvA3EGlP8AiPVgkvVA/BOIeYBYCog/ATEzTHIHEM+Fsh2A+DUQz4JJonuFE40/7AEArk8TP9rbjGEAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAXCAYAAADZTWX7AAAAX0lEQVR4XmNgYGCoZCACjCqioqKJQPwDyrYC4v9ALAHEX+EqgKABKsENxLOA+AUQJwNxP5IasHUgXU5AvASIdwLxOSAWRld0GIjboPxOJDk4ACkSAGIWKB9k7SggAwAAKusTWe+APyAAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAXCAYAAABj7u2bAAABVElEQVR4Xu2SPyhHURSAj0L+hUmECZtMmMhmYTFQShlZSJmkTIoSWS0GAyX6Lf4MbAxKJJtFGSTFoMii+K577nv3vfT7FQPqfvXVOffcd995510RkRn5Y4SGchEaykVoKBf/rqE53MJ6zSfwLi5/m2Kcx12NI7I1NIp7mOetjeOGl/+UKUmdZxpaxE0swx2swxZ8x5p46yc3eIyd+IhdyXLENO5rPILPGjdpXKm54QrHXLIkdkN1VLb04m1qrVZsk22Yjy/YkNgRM4inGh/gK5bipNhf5bOOCy7ZxrO4FtEndho+/Xihcbtkv0tmctc4IHZa5izzzCVWePsMa7jsEnNpz8W+6BAbdb0IH7Bbc8MJDms8hPe4goX4hB1aM5hpmGn2aJ7B1bgcUYJv2OoW3KVO3HRlVr4+xOGeaRY7DXO4T5UXl2OBlzvMBx6lFwOBQOC3+ABytzzFnZKVXAAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAXCAYAAADgKtSgAAAA/klEQVR4XmNgYGCoA2JGKmM4ABlOMzBqOFYwajhWQMjwbiBeAcRSQMwDxD2o0gzqQLwNiPPRxMEAn+E5QLwRTcwHjQ8CbEB8igGLHMjwCUC8DIi5gXgPEAsDsQEQ/wdiEYRSBksgngLEmUB8D0kcBEwYIOoFkAX7gfgzEIsiCwKBPxA/RBMrBmJWIA4H4sNociDX/wJiPWTBdUB8ElkACoKB+C6a2FooPRmIO5AlGCCW/gBiY2TBdiA+D8U7gVgBKs4JxO+A2AHKlwPiP1A2SN0VBogDYCARiK8j8cEAFqEgw9AByOIZ6IJAwMwAcSkMgErC7UCcjiQ2CkbBiAAA+b0oOethLCUAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHMAAAAnCAYAAAA8XHcHAAAEx0lEQVR4Xu2aeaitUxjGH/MY1xgyRFxD10wyy1yGiEyJzOJSJArlJDLdiD/M6nLNrlySIQm5l8xEKOIi89CVDBniec67lv3utb/vnr33+fY9t937q6fW967vnPPt71nrXe9a+wDAJQiGhjBziAgzh4gwc4gIM4eIMHOICDOHiDBziAgzm2dl6uDU3pY6mdqs1d0322OM3xNmNsv+1KvU7ul6Nep3ao98wzhYl3qIurLsyISZvbMIdRo1g3qE2ibFr6XuyTclDqWep06lHqN2bettsTX1ILVTuj4c9jPiXOp+avF0fQz1Iew52ggze+cG2MtdhrqL2ijFP4C9eM811JfU8dQtsPvreB02SMQT1MzUPoN6LrXFxtS/1AYuNkqY2Rt6gb9Ra5Qd5FdqtyI2h5qW2rdRl7m+Epl3EbU+9Qk1O8WfofbMN8Fm5M+oSN1hZm8cR71dBmEv+A+0p9ElU2zTdK3UqDW1Ds3462Cp+mjqY9j9SuWebObeRTzM7JGVqDeod2GpbxPXdzt1p7s+h3oqtZen5sHWzynUrdSNqS+jgfIt7G8IDYQjW93/o9gX1GJlh8yU00snaTQNki1hi30/6Nk6Fv2ECgatN0u52ArUIaj/mfEgc0r02fSStTWpQu9XqJB5k9rH9Qk9+4ruelXX9mimnl8Ghcxcm/oMtoA/CsvXp/ibGkJpRClkrbKjS15C5wvInEB9V8Rkvj601p5JRd+g0HO8QE0uOxxrwqrcXtHgvJ66j1q06Bslp1mNlDNTeyr1VWo3xVnU4xjfLJGZ+5bBhH6/9mBVqBp8sgwOEFWbOihomr2o/cqgR2buTP0CSx27UJ9Sx8JGth8BMkIx5eolUtubo/ShWJbuEZvDSmllgEy3ZbhnfmbOpV6mtoOluoPaeoF3qAuKmNCap2epU1m1qugo75lInQ6HzDwPlmanUyPUjrCXopMLv6BfSP0NM/wB6n1qddevma3yXBWZioE8uw+ALeyemeiuDPfUmam0rcGyA2xw/QRbvzx3oLPg6Ael660WIrUNNpn5MHWFDyZeg72YSbBZppGvklgojSh3ezakvk5tLfZHpbbOKb9J7Uy3ZbinzkwVP6ouhbLAPHSuKRpc2rSX6JmV/uvkB+tCj1KcRvW0soO8CJuFSlFPw2aAN/Pe1M7oxXwPOwy+mVo2xVWlqTg5MF2LbstwT52Zip+U2odRP8CyjE5ohP7+n7DPMtH4ar6JClzFlPwbZX77zJz63qOugv2hKjO14F8OMzOnVnGpa49Qd7vrbstwT52ZJdnEjM4ylWUmmrKab7wCrzNTX7f8CBtFSl2rUEdQ/8COrLTmfUTdBHtRMm4qbM3UQ0+nPkcLFU2zYINiORfvhfXQadRYaLYrBa9Tdixgqqr5xivwOjMHgT6IUm3dNwdNoxR+NvofPP2gz6jU6b9R0UCeC0ufHmU2FZXaO2rG+kylQfsXtYWLjcmCNHPYqTuEr6rmB1KBh5nNUXcIX1XNj6cCryXMbA5tsV5JbVWZ+V9HVHdoFvpjTH2RnY06Eba/noLWDkC8BduLd02Y2Rxao6u+Uanamg1kO/UszP3QYHUx7MRMh/AjFf1e+i8GpeWrK/rqFARBEARBEARBEARBUM1/HP1PxoIFwVsAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIEAAAAnCAYAAAAo5SujAAAFZklEQVR4Xu2ZV4glRRSGjzmtWcwZM0ZUxMiaFQOLiqIorjliQkTxwX1QFFkUfRAVldFFXXPOmDBhRhQUTKsiRkyoiGA4356q6XNru2d6+t6Zh53zwc9Wn647t/fWX3VOVYuIXCLBpCdMEIQJgjBBIGGCQMIEgYQJAgkTBBIm6MoKqoNTe1vViarNqtud2V4G83fq4JlrCROMnf1Ub6impuuVVH+qds8d+mBt1T2qK8obNSyg2qkM1rCy6mbVu+WNTJigmR1VN6nuV52eYlep7hjuYUxTvag6SfWwatfe28Nso7pbqoE7VHVyap+nmq1aOF0fpfpIbKCboO9vZbBgQbG/cYqYcWsJE9Szh+oL1bqqI1VnpfiHYgPmuVL1tWq66gbVrJ67vbylOjW1H1fdl9qnqV5IbdhI9Z9qfRcraWOCTJigA69LNVieP1S7FbFXVDNT+0bVZe5eCUv9xar1VJ+JfRaeVu2ZO4nN3l9l3hSzpepbp3+L6zWrrj2ECcbIkmI/LrPRw8D8Jb3L/aIptmm6ZqWgZmjiGtXVqtvFVphPVPuqHvCdpDLBXkXcEyvBOEPuZ6ayfB/r4tQIt7nrc1VPpvYU1S9iy/rmYgP4Q7qXOUb1nWr5dI2BjqhuD3O46ivVQuUNRxsT8HlWJ9LVP6q7VLv09BAzAa5bPAln9wtLFkVQF/j+pmLoMDFHL+Ziy4gVZk2f6YelyoCyldjgrFjeSPAb5mfht6VG8PDsy7prdhZ1UIxeUAYL+J6DymAXeNA1xIoglrKHxGYAlW4XrhVb6lYvb7SEHLlPGUwcp/peegcc0/Bj8bnlXHw8mS62G9i4iJfwG+Zqvy2YmpRxp1hlPyHkdPCO6ozU5t9vUnssnKl6VPqblSOZgL9PYVUHW7i8LE8EGIADokFDcUiNMKFggp1Vv4vlNPLF56qjfacWkAPZ0qzlYm23Q56RTDBH9ZpqO9WXUp3YZd5TXVjEgAHjWZq0WtV1Lju4e/OrenY+mOB8sXQwpJohdkiSWUJsgD3MdAoXqtvMAWIFj6ftdsjTZALSCyZjgPj+n8Tys+cW1XVFrAtTVFvP51pVHJiAGXq5Dzo2FDvl8lyvekbMMBlmJftUT9vtkKfJBBSFH6T2FmJVeJkzORqlEi7ZQCxNNWmVquvkhGWaGTaziLMXfln1turH1PYnYZyazXDXVL0UbQe6WNvtkKfJBK+qTkjtQ8SeaUhspQK+/2+p2f6MM21eJE2T3h1Bv7CK+u9gR8LZhofnyrXZOjJv6oSfc2O0c4K6lQBKEwDXzPpM2+2Qp8kEJXnwM5y1v1nExhsOhTiAmZqu+f/VvUjaWywPTy3idbD1LA+pSoakyumkck43qek2EUtljNcc1fOpD6n6udTOYKI5+aKLCdiq8UbqfbF9e4aDiQfFztLr9thtwLXlAI8GqwvP4ovS8YYzfYpT/52kFVY7ZmYJ/TlfaDrWzWBm0tpIDImZgN87r7zUXLzfYPZT6LIi8XxQZ4L9xd6LzGU0E/BHcVdb6M+DNb1JGzSkGv7zXU3XhkG9TaQWurSIlYzFBJ6XVBe5a04GmYxQZ4IeRjPBZGeQL5JIH+UxMrCDoqhGFLyklHz9hOuXGZJ5n+kRse0xh2fUcaSgTJigDyi2BvkiiZ3RcDHWQNeVgJXnHLFtcjmmYYI+IMUwsNQpbEePl6ri/lh1dmoDKYOzC1hazDzUC9kUwHLNgdZIdDXBY2IrgH+5lQkT9Mkg3iZm7hV7tzISXU3wlFhKWKSIw6gmeFYsJ4XGLgYVg3AgVt7z4tCME9nZNfe6iBnPLqyMN+lW1ac1cRQEQRAEQRAEQRAEQRAEFf8D8D2Xv9rQKwQAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAXCAYAAADZTWX7AAAAdElEQVR4XmNgYGCoZCACjCpiWA/El4D4DhBzArEiEG8D4o9AHAZTtAyIWYD4OxA7ArEQFIMUNsEUway7CsRBUHYJA0QzHMAUXQRiXyCWBeKbQCwCV8GAUHQeiH2A+BgQOyGkIQDZul5kCWQAU8SNIjoKSAYADIQV4VviHrUAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAXCAYAAACf+8ZRAAABXklEQVR4Xu2TPyhGURTAj/8SxaAsEgbl32i12UzKblAWSSkpGYiJxWCwSUrIggUpZSVSipJkxGIxSPmd79333vlufcv31Re6v/rVPefc997pnvtERObkDxKaLhah6WIRmi4W/6LpRnzCES+fD9N4ipc469WUSfz0XHO1a5P7wnGXz+A3XYof2OHl8+EVa7ANv7EzuyyLOIqtzivsc7UH7MYWZ63LZ9CmF3AX63ADD+wGwzHe5XDY7IvZNOsLXDKxj35zyMT3uIeHOCXRYSYsSzSCJhfv4ExaLogtsz7BeRNbevERy0yuGeuxB18kumoJ23jr1jrKd+xPy1nofdvP4YDZF6OTidF7PWZii/Yw4ScN+l098YQVfMZzHMQ3PMMuuylP1iV6r74v/pG0OT3VmFWJpm1pxxuJGtUJHWGD3aB3ulzS0VRjSVoumCrJHrtPheSu67OVfjIQCAQCv5sfH/VHzUYGWyMAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAAgUlEQVR4XmNgYGCoZCABjCpGBiQr1gHi/0CsiCYHAolAfBjGASnWAuLHcGlUMBOIY2AckGKQwB0gDgLik0CcDMUgIAWlwQCk+CIQl0D5p4DYE4oxQD0Q/wZiESDmAuIfQCwIxRhgOxAvhbL1gPgDEC+HYj6YIhhADzpONP4ooDMAALIgFzrE6DSjAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAXCAYAAAA/ZK6/AAAAlUlEQVR4XmNgYGCoYiARjGogBpClgQuIlwLxOiA+CcSXUFSgAZCGUiBeAeULAfEahDQmAGmwAuIfQDwHiFlRpTEBzA/mQPwUiOci8T8AMSeUDwfInvYB4tdQtgwQnwbiUCAOgKsAgpdAfAOIVwLxUSDWgIp3AXExEItD5eEAFkrcyIJAcAiI1YE4iAHiTHZU6VEwkAAAQrQaNtOyPCIAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAAA7ElEQVR4Xu2SIWtCURiGv21FWZhFDeIwCCaLyWQXDNa1LfsHLGqyWFYV87IIRhGUgdrEX7C6ZloTfQ7nO3K5qHhBls4DD/fwvQfewzlXRKQl/4AviYQvicQ9S9LYwQE28dEFpuQB3/EXUy6AF5zgCLdYCmSXSOAbTjHuhm0c4xAPusnR1bnBnHCl6wouQn5qVsa+2L1JnZ2uKyu2xJzescOqrvNic/O9RB2/sYczCRz4Wskf1nSdEZvnTul5niRwTY5rJT9i38pQxD3GXBgFU9LAtdiSOX5o9oob/MIlFnQemVt+4efwwOPx3JcjUcYrjmgkOiAAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAXCAYAAADZTWX7AAAAfElEQVR4XmNgYGCoZCACjCoCK9IC4pdAvAsqxgLEh4F4P7IiELgJxL+AWAzKnwPEPlA2WFEQEHcDcQUQ7wZiTiC+DlMAU9QJxOFArALEv4HYG4iXoisC2a0A5Z8H4qNAnA9TAALbgfgTEGtD+eoMELexw1UwEBkEo4AwAADCYBiAH5cqpgAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAANwAAAAxCAYAAACxgMfdAAAHiUlEQVR4Xu2cB4zcVBCGB0LvBAgd0bvoHURAoXeQKKEeondRRScgEEiIjqiBBNE7iF6CTgIkWoQCSLQAAULvvZf5Mn7nt+/s3cvd3t7t7nzSr9h+vt31ezNv5o3tiIicJY7jNAx3OMdpIO5wjtNA3OEcp4G4wzlOA3GHc5wG4g7nOA3EHc5xGki7O9wRqnGqOVXzqR5Qjag4w3HqSLs7XIdqsuqwbL+zq8Vx+oF2d7hrVAepXlHNrbqrstlx6ku7Oxzp5LSq91UnqY6rbHac+tLODney6tlse0HVT6oZ8mbHqT+t5nDD0gOOM5hoFYdbQHWTakLa4DiDiVZwuFVVj4gVPMYnbY4zqGgFh5smU6g2FkFhZKJYcWRqtZY4Tp1oBYcLVHM4wOmeUv2nujNpK2Im1a+q4WmD4/SWdnI4oKjyqZjTHZK0FXGBauX0oOP0llZyuCOlZ0UTItbfqt9UqyRtjtOvtILDTae6XvWJ6h+x4snGFWd051SxKPematakzXH6jVGqOcTWKwG2w7Eh2TZiO2b27Hj8t80CRZbHxJyO2wk9ZWnV4aoN04YWIowrop8CsS2kk1Q1OxlsMEGPVM0m3e03tn0os/FwHp8Rs6tqaHKsAiIcJfUfVOdlx3jaggd6A/erLon2A3uKRZVmBsf5S/WzavmkrYwbVOenB1sIxvXjbBuHe1l1eraPLVyUbS+p+lfyfiuzk8EETxbtF+3Xsv0yG8dpmaw3SBuUxcX6b7Hk+BRwuIfEUqw/VMuKfdhH0Tn3St7JMXtIPjBlcAF8cZn4roGGZyjpvNdUMydtRaQO1yGWxtJH8ZqQIs59ka6N2g4Qi7DLqE4QM9bVovaBJB1Xxv+2aDvYAmPLZLVzQVsZc0l3Gwgqe0poDbE+v0e1j9hn3C7Wp2QcR4l9N45/nVjG0qF6WnW05NEJ52DdHttcLdtP+yJAxRubWS9tyOD3FE7KweEYbLz9SbEPq5fDcREvVtHC+alTDe+vUZWspge7zi6HWZw+oAPDazrViB2O1ORdsSdd1lf9oppFtbbYIJJebCp2e4G3EYDnNnkPr1P1kmpbsTVoMGoYyFsRYVyZfEaovlPtmLVhC0SxeVRnqD6TPIUqs5MYrju1gaCbo/MCOOFXYvdCsRX6dDnV9mLPvi6qulpsAoNtxCaBg8XW8USrQ7O2c1QvZNuBWrZfZuO1HA4nfjU9CMHhmEWWUv0ulod+GJ1T1pFlP6av0Cmfl4hKZGB61eo1tGLX2dXhmrl2nlqpRexwzLS84hOYJDbj76D6VmxgFhFzuHTd84Hq2Gyb6+IeIWDAGMBAwbiSYo9RXazaJGrDFt4Q64MzxQw+biuyk77AuJDyYcAI5yfKwYWq97J/A0xuTAIBfidjBGNVV+ZNU6hl+2U2XsvhtlR9kx6E4HBrZvtni30hT2UE0o7cTmy2Tn8M4Z1ZPoZjj1bRQvmpAwYR5wux2bcnxA6HY2CUgfFi9/dYUL+jukUs3Tw3Ogf4Tm5NkB4BqRCfSbQlxd06Oz4QpOMak9pCTNoW7CRmX+luA0FXROcF9hdzcOwoiLfzYSuxFDHuWxwu/u306RPZNhG0yOGq2X7aF8HGixwuTASwher7aL8LwjGeODo6xuwcwiprki/FjIccmujDF2EonWIzN8f54YT7A6f8VWOhUoSBh4oakY/9NKIUMUos0gTDr8USYoPylmpesbSWNcPjYmsynAWYNekPBmuS6nmpnIxwUlIl+p+0d6eorVPs72Ez1dd5U78zROz7GeOOipbcFiapVqhsKrUTxqGvnKZ6W2yNfavYdzNmN4qNMdtEQdJ4HI6JjInwYdUzYpVGwAlI+blGIAOqZvuhL4psnLUh18fnM9EwzsGx4W7JJxDO7YqaRLgiZkwPDGJY+3DxYZ2xkupHKb+2AIPDDLlO2tALWJgHh4djxIyDYgCz4lipXKNhkLuJrfdiGGQcMRgJaRvRr92hX1hT1iJEOOw3nXAZn9clL/KU0Vfbn1/saSaKON2oZZTNwCli67tg0HQsKSJOUMYwsU7prze8WddReWSNQ/pIBKRqCRjOn1Ls6PxmIgXpD9GBxT8R2+kZu4tFu7Kx51YGKeouaUOdoGZANjMybQi0gsPxP22dKLbQJ2JQxXqu4oxKcEjCPylCHJXqCbMra8JLxe7xbB61kYoSuVjPFIFD9tfvanWYQOlbqpVlkD0cL91vWtcDqtxkNKW0gsORc+NoRAaqTB1SWblKISKSdlDarsW6qo3Sg47TW5rd4fYSiyLAfRei3AQprn7iYDgn5eWeQKSk0sR6zHHqQrM73FWqvbNtSrWTpfI+SoAUjRSS4grtVLaqaZLYg9CcH5d7HadPNLPDUYqmrHtZdIynIO6I9gOUjKlc9kbcxHScutDMDtdbhqcHHKdRtJvDDZWePTZFOZ90NdwkdZy60G4Od7nkj3CNEbt9EGtE1gbcz/HyvFNX2s3hOiV/bIonCrjnFSuOaO5wTt0ZJ/ZyYTuI10m4bcBrGuzzSsjERKOj83nsi79JP8fl6o0cx3Ecx3Ecx3Ecx3Ecx3Ecx3Ecx3Ecp7n5HzBaJm8gTJO5AAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAXCAYAAADHhFVIAAAAXklEQVR4XmNgYGAoZ8ADhpmkCRD/B2JRNDmwpDEQ30OXAAGQ5AIgLgBiOSC+A8SWyJLXgXgREB+DKoCDeiD+BcTJQHwbWQIEdgDxXCh7IRDfAGIfmCS6VzjQ+MMeAAAqKhLekcvedQAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAXCAYAAAA/ZK6/AAAAiklEQVR4XmNgYGCoYiARjGogBgxSDX1APBuIOYH4EBB3oqhAAz1A/AmI+aH8U0AcAMTiQHwJiD2g4nCwAoiPQtmSQPwHiCWg/GtALArEqUDsDRVj6Abix0C8F4gnAfFyqDjI5KtAfIcBYhscgPzACsSMQLwAiPOg4g1AnAPEx4FYB4hZoOKjYBAAALQkGK5umhS8AAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,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>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAQCAYAAADAvYV+AAAAXElEQVR4XmNgYGCoZCABDCLF6UC8C4inAjE3EDsA8XYg3gzERnCVQDANiPcCMScQO0EVvAPiACj/OkIpA8M8IN4ExExAHALET4B4ORCzADEfEN9FKCXDzaOAtgAAFBUPtSrX4BQAAAAASUVORK5CYII=>

[image31]: <data:image/png;base64,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>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAXCAYAAACvd9dwAAABtUlEQVR4Xu2UPSiFURjHHx/5LvkYGMRmQRYWFkkWpWQx2pTyNcggXUSUCeUzC7HYJIn0+hgUigxYENlksxH/p+e87jnHveoqw6vzq1+35zzn3Pc+z3nuS0Q0QP8YV1xQccUFFVdcUAl6cd1wRjNHT47CdXgAT+ApLNQ3KHrhMVyFl7DeTMdMKfRIvleH4114RrE1vh822IsTcBMmqviWZKNOOXyHeSrugVfh9BcdMNNaa4Rl1hrvW4aPJJ334YKfYRosInlmicotkVyAbpXKZZE06hvcnVQYgjdw2MgKCSQ3tgOH4DWsNnaE4WZx19tIfnyxmTbwYJcWz8EVLeYCxrU4Etkkz4u3EwwXlwSb4Cx8hbXGDrmNC9gH6+A5HDR2mGzAD4reAB+PzOIWSZrosw1HtDga3PyI2HN9CNestRZ4p8XN8EWLdVrhPZwmaUiukTXxyCxujGQ6fPgd0K7FMTMFH0j+d1vqk2c4n6T7FSSdmYRHcIHkxVLJhzXiVK7AWg/BGmuNz3ID3+ATycSkkIzWPNyHe7DTP/Bb/JvLIBnPn+CHp9uLf0Ay/TBqDofD4XA4iD4BcFNT3gya6j4AAAAASUVORK5CYII=>