Default output root for src/data-pipeline/data_pipeline.py and data_orchestrator.py.

Each corpus directory uses the pipeline layout:
	<corpus>.db          Document-preserving parsed records, where available.
	<corpus>.db.sha      Pipeline backing-file checksum sidecar.
	train.txt           Existing training sentence export.
	test.txt            Existing test sentence export.
	rare-ngrams.csv      Historical diagnostic report, not another data split.
	cleaned/            Retained cleaned files, where supplied.
	cleaning_audit/     Retained audit files, where supplied.
	archives/           Retained original compressed database, where supplied.

On 2026-10-26, seven supplied corpus directories were moved here from nlp_corpus.
Legacy split/report filenames were normalized; all 41 moved files were verified
against their original sizes and SHA-256 hashes. See corpus_import_manifest.json
for the old-to-new mapping, hashes and decompressed-database provenance.
No sentence content or train/test membership changed. No valid.txt was supplied.

Available document stores: blog-posts (19,320 records), gutenberg (18),
inaugural (60), movie-reviews (2,000), reuters (10,788), webtext (12).
ROCStories has sentence exports but no document database. WebText's 12 records
include raw/cleaned variants of six sources, not 12 independent documents.
These counts do not establish disjoint document membership in the exports.

Original blog-post.db.gz and reuters.db.gz files are preserved under their
respective archives/ directories. Verified decompression produced blog-posts.db
and reuters.db; all six available stores have verified pipeline checksum files.

The stores have SQLite-backed Dict(key, value) schemas. Python 3.12 dbm/shelve
does not recognize this supplied format. A compatible interpreter or a tested
read-only SQLite-shelve adapter is required before document preparation.
Successful filename/checksum checks alone do not make shelve reads compatible.

Do not regenerate these stores or overwrite the supplied splits to fix that
compatibility issue. Existing --load and split stages can replace artifacts.
The experiment suite must read prepared data without automatic rebuilding,
re-splitting or downloads. See docs/research_directions/data_efficiency/plan.md.

Git contains this guide and corpus_import_manifest.json, not corpus contents.
A fresh clone cannot prepare or train the real suite until the exact imports,
database checksum sidecars and provenance files have been restored privately.
The Colab bundle is also private and excluded from Git; it contains the audited
sentence indexes needed for exact training/validation/test reconstruction.
Use colab/README.md for packaging, notebook execution and complete retrieval.


