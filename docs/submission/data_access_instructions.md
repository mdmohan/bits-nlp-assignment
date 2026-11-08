# Data Access and Exact-Replay Limits

## Distribution Boundary

The group ZIP maximum is 10 MB. Raw datasets, checkpoints and large scoring-event
archives are not shipped. Code/docs/selected small results are distinct from the
private complete workspace. Public availability is not redistribution permission;
review the [dataset card](dataset_card.md) before acquisition. These are recorded
source URLs and byte pins, not downloads or live access checks performed for this doc.

## NLTK Sources and Byte Pins

NLTK 3.10.3 is recorded. The `gh-pages` URLs are mutable and no upstream commit
pin is retained here. Archive SHA256 identifies the obtained version; current
branch contents or the NLTK version alone do not. A different download is a
new input, not exact replay. Seek a matching lawfully retained copy; never bypass
hash checks or invent a release date.

| Corpus | Actual public archive URL |
|---|---|
| Brown | [brown.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip) |
| Gutenberg | [gutenberg.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip) |
| Inaugural | [inaugural.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/inaugural.zip) |
| Reuters | [reuters.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/reuters.zip) |
| WebText | [webtext.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/webtext.zip) |

| Corpus | Recorded archive SHA256 |
|---|---|
| Brown | `9b275f9b3b95d7bd66ccfb7cd259f445a13bbe5d1f4107aba09fd3e8364bafa6` |
| Gutenberg | `2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611` |
| Inaugural | `b3b2cb7bd82697b58ff79c75af20ab963810d191a8c3d1f7c240f98e5d27a895` |
| Reuters | `9a59a43823f02a6e2777075c989a3dc454e4b6f68e0332ee3c0e8264075b62f5` |
| WebText | `9e32dbae4879464b8f420a0dc721855bb26167b720d7695588d2ca2aeadf501a` |

Full-workspace provenance:
[Brown](../../results/comparison_refresh_20261102_v1/processed/brown/source_manifest.json),
[Gutenberg](../../results/comparison_refresh_20261102_v1/processed/gutenberg/source_manifest.json),
[Inaugural](../../results/comparison_refresh_20261102_v1/processed/inaugural/source_manifest.json),
[Reuters](../../results/comparison_refresh_20261102_v1/processed/reuters/source_manifest.json),
[WebText](../../results/comparison_refresh_20261102_v1/processed/webtext/source_manifest.json).
Each records archive path, URL, upstream checksums and document identities.
General access information: [NLTK Data](https://www.nltk.org/nltk_data/).
These private provenance links need restoration if omitted from the group ZIP.

## ROCStories Mirror

Use [hamishivi/ROCStories](https://huggingface.co/datasets/hamishivi/ROCStories)
revision `cd2b3893a6ee30afada293d5a8f877c3023530e1`. The combined 2016/2017-release
mirror's card declares CC BY 4.0, not independently certified upstream rights.
Resolve upstream terms before redistribution. Original project information:
[ROCStories project](https://cs.rochester.edu/nlp/rocstories/).

| Pinned public file | SHA256 | Recorded Parquet bytes |
|---|---|---:|
| [data/train-00000-of-00001.parquet](https://huggingface.co/datasets/hamishivi/ROCStories/resolve/cd2b3893a6ee30afada293d5a8f877c3023530e1/data/train-00000-of-00001.parquet) | `dd094ab7a28ae1b38ce4489c94e861d62cee015ca452589da17a3fa3c55f913d` | 21,011,582 |
| [data/validation-00000-of-00001.parquet](https://huggingface.co/datasets/hamishivi/ROCStories/resolve/cd2b3893a6ee30afada293d5a8f877c3023530e1/data/validation-00000-of-00001.parquet) | `8b926a61bc1af0ea4d6f1d94eac6a6e8ed24554c6c64a7ed96f9fac29b9a79d3` | 2,335,541 |
| [data/test-00000-of-00001.parquet](https://huggingface.co/datasets/hamishivi/ROCStories/resolve/cd2b3893a6ee30afada293d5a8f877c3023530e1/data/test-00000-of-00001.parquet) | `f5a0c34c859b834ef60bdae387df3f673a7523a01786587eefc0f391c8ee7969` | 5,841,339 |
| [README.md](https://huggingface.co/datasets/hamishivi/ROCStories/resolve/cd2b3893a6ee30afada293d5a8f877c3023530e1/README.md) | `d42042693434f50515357a39a81965137c9155f508392cae617e92a502f630b0` | Not separately recorded here |

The [download manifest](../../results/comparison_refresh_20261102_v1/raw/roc-stories/download_manifest.json)
records the upstream tree/LFS IDs, sizes and hashes. The locally assembled
`raw/corpora/roc-stories.zip` SHA256 is
`a7a0d34ce0952eb452441674e2e5491ef6761072e2a9d09a90e1452aefa68ae3`;
this is not an upstream ZIP URL. Re-zipping identical members can change
archive bytes. Preserve the original ZIP for archive-level replay, or verify
member hashes and describe the changed container honestly.

## Restore Matching Inputs

1. Obtain sources lawfully and verify byte pins before extraction. Keep secrets
   out of code, provenance and archives.
2. Restore `results/comparison_refresh_20261102_v1/`, its frozen configs and
   `processed/<corpus>/{train.txt,valid.txt,test.txt,prepared_manifest.json}`.
   Retain source manifests, document/split membership, occurrence/quarantine
   provenance and leakage/near-duplicate audits separately. Hashes alone do
   not recover missing text, order or exclusions.
3. Match prepared-file SHA256 values and settings. Brown/Gutenberg use 80/10/10;
   Inaugural/Reuters/ROCStories use 70/20/10; WebText uses 20/40/40 (one/two/two
   source files). The split seed is 42 except for WebText, which uses 13. Preserve
   test-first exact quarantine and the recorded, unequal near-audit scope in the
   dataset card.
4. Match frozen code/config identity. Relocate runtime paths explicitly because
   saved configs contain original absolute paths. Do not substitute mutable
   downloads, flattened ROC exports, pilot splits or changed scientific settings.
5. Saved-model/event replay also requires matching V3 artifacts under
   `results/comparison_sweep_20261102_v3/full/runs/`, omitted from the group ZIP.
   Deserialize only trusted signed checkpoints/pickles, when explicitly authorized.

The private frozen V3 bundle SHA256 is
`2ae43095c6e7f0fc95468b13a40c9d49b9004c7579bb5b961b2f7f6e60bbfff8`,
with 8,492 member hashes. It exceeds the submission ceiling and is not the
final source/group ZIP. Public sources permit attempted reacquisition, not
guaranteed byte-identical preparation: original tokenizer resources, cleaning,
segmentation, document order and quarantine also matter.

## Explicit Replay Limits

Small submitted tables/figures permit inspection/arithmetic checks. Metadata
analysis needs the expected run CSV/JSON layout. Accuracy, Laplace and event
category/cohort audits need omitted events and, where applicable, prepared
text/vocabularies. Inference needs omitted signed weights/metadata. Statistical
comparison fits counts and is not read-only replay.

Thirty shared count passes supplied 180 statistical conditions, but only 20
serialized count archives remain; all ten Gutenberg/Inaugural archives are
absent. Saved metrics/events remain auditable without those counts; count-state
replay needs an explicitly authorized new counting pass. Original neural event
context coverage is null/unavailable; later audits reconstruct window/exact
prefix membership without model inference.

No hosted public full-run/bundle URL or guaranteed external reviewer access was
supplied. Fresh-machine replay, complete data access and live URL availability
are not certified. See [reproducibility](reproducibility.md) and
[source commands](../../src/README.md).
