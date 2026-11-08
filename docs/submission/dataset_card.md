# Dataset Card: Final Six-Corpus V3

## Identity and Intended Use

Group 1, Problem Statement 1; documentation prepared 7 November 2026. This is an
English word-level next-token study, not a released general-purpose dataset,
story-reasoning benchmark or cross-domain transfer experiment. Brown is the PS1
anchor; the other five corpora are supplementary. Sources were obtained on
2 November 2026 according to recorded `accessed_at` fields. Exact UTC times,
public URLs and byte pins are in [data access](data_access_instructions.md).

The authoritative preparation snapshot is
`results/comparison_refresh_20261102_v1/processed/<corpus>/`, not the older
imported exports or whole-document pilot. Corpus/category tags describe sources;
they are not learned prediction labels. All six sources are English.

## Composition and Sizes

Counts are post-quarantine. Documents are source units; sentences and word
tokens are occurrences, not independent statistical replicates. Words exclude
BOS/EOS/PAD. Nominal ratios differ from realized fractions after integer
rounding and sentence removal.

| Corpus and content | Documents train/valid/test | Sentences train/valid/test | Words train/valid/test | Nominal ratio; split seed |
|---|---:|---:|---:|---|
| Brown: full 500-document, 15-genre corpus, not press-only | 400/50/50 | 44,766/5,489/6,577 | 924,218/114,416/116,408 | 80/10/10; 42 |
| Gutenberg: 18 literary/book files | 14/2/2 | 49,356/8,605/38,181 | 951,248/201,507/978,972 | 80/10/10; 42 |
| Inaugural: 60 US presidential addresses | 42/12/6 | 3,877/1,111/406 | 99,657/30,361/10,934 | 70/20/10; 42 |
| Reuters: 8,212 retained parsed news documents | 5,748/1,643/821 | 41,022/11,525/5,907 | 889,683/250,690/126,774 | 70/20/10; 42 |
| ROCStories: 98,161 narrative source rows | 68,710/19,635/9,816 | 339,616/97,538/48,884 | 2,980,263/853,632/427,651 | 70/20/10; 42 |
| WebText: five heterogeneous retained files | 3/1/1 | 6,398/17,958/316 | 53,269/156,464/3,550 | 70/20/10; 3999 |

Count/settings authorities:
[Brown](../../results/comparison_refresh_20261102_v1/processed/brown/prepared_manifest.json),
[Gutenberg](../../results/comparison_refresh_20261102_v1/processed/gutenberg/prepared_manifest.json),
[Inaugural](../../results/comparison_refresh_20261102_v1/processed/inaugural/prepared_manifest.json),
[Reuters](../../results/comparison_refresh_20261102_v1/processed/reuters/prepared_manifest.json),
[ROCStories](../../results/comparison_refresh_20261102_v1/processed/roc-stories/prepared_manifest.json),
[WebText](../../results/comparison_refresh_20261102_v1/processed/webtext/prepared_manifest.json).
These full-workspace evidence links do not promise that all private provenance
is shipped in the lightweight group archive.

Brown's test covers 14 genres. Gutenberg tests on Austen's *Emma* and the King
James Bible, validates on Chesterton's *The Ball and the Cross* and Whitman's
*Leaves of Grass*. Two books provide narrow, dependent test coverage, and test
text exceeds training text. Inaugural tests on Washington 1793, Monroe 1817,
Taylor 1849, Buchanan 1857, Truman 1949 and Carter 1977.

WebText excludes `firefox.txt`: training uses `grail.txt`, `pirates.txt`,
`wine.txt`; validation uses `overheard.txt`; test uses `singles.txt`. One test
source and an unusually large validation source make it a source-shift stress
case, not representative web-language evidence. Reuters uses the prepared
parsed collection, not every item in the original NLTK archive.

ROCStories uses the public combined-release mirror, not old flattened exports.
Mirror Parquet partitions are recombined and repartitioned. Original partition
plus zero-based row supplies source identity; identical complete-story hashes
are grouped into one new partition. `text` is checked against `prompt` plus
`continuation`; segmentation preserves all non-whitespace characters. The
neural sequence unit remains the sentence, not the complete story.

## Preprocessing and Representation

The [preparer](../../src/prepare_mega_run.py) freezes parsed documents,
membership and sentence occurrences. NLTK 3.10.3 is recorded. Corpus-specific
cleaning/parsing precedes model tokenization: Brown is uncleaned and retains
punctuation/POS forms such as `,/,` and `./.`; other NLTK corpora use configured
cleaning paths. ROCStories validates pinned text without that cleaning stage.
Frozen code/manifests, not a generic tokenizer label, define exact bytes.

Model input is one nonempty sentence per UTF-8 line, lowercased and split on
whitespace. Attached punctuation remains part of token identity. A sequence
is BOS, words, EOS; state does not cross sentence boundaries. Reserved IDs:
PAD=0 (`<pad>`), UNK=1 (`<unk>`), BOS=2 (`<s>`), EOS=3 (`</s>`). Prediction
excludes PAD/BOS and includes UNK/EOS. PPL includes EOS; word accuracy/OOV
exclude EOS. OOV correctness scores UNK, not recovery of the original spelling.

## Budgets and Vocabulary

The five 20/40/60/80/100% budgets take the first `ceil(N * percent / 100)`
nonempty TRAIN SENTENCE LINES after sorted document-ID concatenation. They are
not whole-document budgets, random samples, exact word fractions or compute
budgets; a boundary can fall inside a document. Validation/test stay fixed.
Brown adds genres, Inaugural is chronological, Gutenberg follows alphabetical
source IDs; Reuters/ROCStories follow IDs without a verified chronological
curriculum; WebText follows source-file order.

Vocabulary is rebuilt from each prefix and shared by all models/seeds at that
budget, without held-out words or a vocabulary cap. Brown uses `min_freq=2`;
the other five use 1. IDs can change across budgets. Threshold-one corpora have
no positive sub-threshold UNK targets in full training text; softmax still
updates their UNK parameters. Changing vocabulary support, mapped contexts,
composition and capacity confounds simple cross-budget PPL comparisons. Fixed
target cohorts fix membership only. PPL across these tokenizations/corpora is
not a universal common-scale benchmark.

## Integrity and Retained Overlap

Test-first exact-sentence quarantine uses lowercase/collapsed-whitespace
identity with precedence test, validation, training. All six saved audits
report zero cross-split exact sentence overlap afterward. This does not prove
zero semantic overlap, zero within-split duplication or independent documents.

| Corpus | Quarantined sentence occurrences | Retained near-audit disposition |
|---|---:|---|
| Brown | 5,801 | No candidates above screening threshold |
| Gutenberg | 624 | Two train-test pairs involving Biblical quotations |
| Inaugural | 17 | No candidates above screening threshold |
| Reuters | 1,518 | Six cross-split pairs: three train-test involving two distinct test sentences, two train-valid, one valid-test |
| ROCStories | 3,192 | Near scan not run; absence cannot be certified |
| WebText | 74 | No candidates above screening threshold |

Screening requires at least 10 tokens, five distinct token five-shingles and
Jaccard similarity >=0.9. Shorter, lower-similarity and semantic matches fall
outside the claim. Flags remain in the leakage/near-duplicate audits, not removed
data. The scan scopes are unequal.

On 7 November 2026, the group confirmed instructor approval for supplementary
corpora and reporting disclosed retained-overlap flags. No written approval
record, instructor name or approval date was supplied. This group-confirmed
disposition does not establish statistical independence, zero near overlap or
universal compliance solely by disclosure.

## Rights, Ethics and Limits

| Source | Rights/access qualification |
|---|---|
| Brown | Not public domain; review package/upstream non-commercial-use terms |
| Gutenberg | Literary/public-domain descriptions do not establish worldwide rights to every file or edition |
| Inaugural | Historical public speeches; verify package and applicable source terms rather than assuming a blanket license |
| Reuters | News copyright/distribution terms apply; NLTK availability is not unrestricted redistribution permission |
| WebText | Mixed sources require individual terms; no common license or privacy certification is asserted |
| ROCStories | Pinned mirror card declares CC BY 4.0; not independent upstream rights certification |

Historical/news/web material may contain personal names, biased, offensive or
sensitive text. No universal no-personal-data, consent, representativeness or
privacy-audit claim is made. No pretrained language model is used; models are
trained from scratch. Fixed source partitions, unequal compute, retained
overlaps and single neural seeds for ROCStories/WebText limit inference.

The 10 MB group ZIP excludes raw datasets, checkpoints and large scoring-event
archives. Review public rights before acquisition and retain matching private
provenance for replay. See [reproducibility](reproducibility.md).