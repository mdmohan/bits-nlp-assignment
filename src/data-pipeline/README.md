# NLP corpus data pipeline

`src/data-pipeline/data_pipeline.py` runs the corpus preparation stages used by the language-model assignment. Run it from this project directory with the Python environment that has the project dependencies installed.

```text
python src/data-pipeline/data_pipeline.py --corpus NAME [--in PATH] [--out FOLDER]
                            [--download] [--clean] [--load]
                            [--split-ratios TRAIN TEST [VALID TEST]]
                            [--rare-check]
```

Choose one or more processing stages. When several are selected, they run in this order:

1. `--clean`
2. `--load`
3. `--split`
4. `--rare-check`

Processing defaults to `data/processed`; download mode defaults to `data/raw`. Processing outputs are placed in a subfolder for the corpus name:

```text
<out>/<corpus>/
├── cleaned/              # One cleaned .txt file per retained source document
├── cleaning_audit/       # Cleaning and read-error reports
├── <corpus>.db           # Parsed document records (shelve database)
├── <corpus>.db.sha       # SHA-256 checksum for the database backing file(s)
├── train.txt
├── valid.txt             # Present for a three-way split
├── test.txt
└── rare-ngrams.csv
```

Only stages selected on the command line run, so files from unselected stages are not regenerated.

## Download through the pipeline

In download mode, `--in` is the YAML manifest path. The default download output is `data/raw`; override it with `--download-out`. From the project root, `--in data.yaml` resolves to the manifest in this folder if there is no `data.yaml` in the project root.

```bash
python src/data-pipeline/data_pipeline.py \
  --corpus brown --in data.yaml --download
```

When combined with processing stages, the pipeline downloads first and passes the downloaded corpus folder to cleaning or loading. NLTK ZIPs are extracted under the raw root when needed; the corpus folder is resolved using the manifest's `extract-path` (default: `corpora/<package>`). If you use `--split` or `--rare-check` with `--download`, include `--load` as well. `--out` controls processed outputs; `--download-out` controls the raw download location.

```bash
python src/data-pipeline/data_pipeline.py \
  --corpus brown --in data.yaml --download --load --split-ratios 0.8 0.2 --seed 55
```

## Download a corpus

`data.yaml` in this folder is the corpus registry. Each entry stores download settings plus documentation fields for comparing candidate datasets. Install PyYAML if it is not already available (`python -m pip install PyYAML`).

Download a configured NLTK corpus by its registry key. The default output root is `data/raw`:

```bash
python src/data-pipeline/data_download.py brown --conf data.yaml
python src/data-pipeline/data_download.py webtext --out data/raw
```

The old `--corpus NAME` spelling is still accepted. NLTK stores its package under `<out>/corpora/`. For a URL-based corpus, add a registry entry with `download.type: url` and `download.url`; optionally set `filename`, `sha256`, and `extract-path` for ZIP content. A URL must point directly to a downloadable file. ZIPs are extracted when the download is combined with loading, or by the orchestrator when it finds a local archive.

Corpus entries may also define `ignore-files` as file names or paths relative to the resolved corpus folder. The orchestrator forwards these entries to the loading stage, which excludes matching files before parsing. For example:

```yaml
extract-path: corpora/gutenberg
ignore-files:
  - README
```

Example command using a config file in another location:

```bash
python src/data-pipeline/data_download.py my-corpus \
  --conf src/data-pipeline/data.yaml --out data/raw
```

Use the registry's `status` field to distinguish candidate datasets from the corpus selected for final experiments. After choosing, record the exact release, access date, license terms, file counts/size, preprocessing, split, and known limitations. Re-check redistribution terms before committing or submitting raw or processed data.

## Full pipeline

This example cleans WebText files, parses the cleaned files into a database, splits by document, and analyzes rare bigrams and trigrams:

```bash
python src/data-pipeline/data_pipeline.py \
  --corpus webtext \
  --in data/raw/corpora/webtext \
  --out data/processed \
  --clean --load --split-ratios 0.8 0.2 --rare-check \
  --seed 42 \
  --remove-word urlLink \
  --remove-regex 'https?://\S+'
```

When `--clean` and `--load` are both selected, loading reads from `<out>/<corpus>/cleaned/`. If `--clean` is omitted, loading reads directly from `--in`.

## Run stages separately

### Clean text files

`--in` must be a folder containing `.txt` files. Cleaning writes retained files to `<out>/<corpus>/cleaned/` and reports to `<out>/<corpus>/cleaning_audit/`.

```bash
python src/data-pipeline/data_pipeline.py \
  --corpus webtext \
  --in data/raw/corpora/webtext \
  --out data/processed \
  --clean \
  --remove-word urlLink \
  --remove-word nbsp \
  --remove-regex 'https?://\S+'
```

`--remove-word` removes a literal whole word, case-insensitively. `--remove-regex` removes matches for a Python regular expression. Repeat either option for additional terms or patterns. Matches are replaced with spaces.

The pipeline uses `--encoding auto` for cleaning by default: it tries UTF-8, then uses `charset-normalizer` if installed, and otherwise falls back to CP1252. Override this with `--encoding ENCODING`, for example `--encoding cp1252`.

### Load files into the document database

Loading requires `--in` to be a folder. Without `--clean`, it parses that folder's `.txt` files directly:

```bash
python src/data-pipeline/data_pipeline.py \
  --corpus gutenberg \
  --in data/raw/corpora/gutenberg \
  --out data/processed \
  --load
```

The database is written to `<out>/<corpus>/<corpus>.db`. The `--corpus` value selects the parser when it is one of `gutenberg`, `brown`, `ptb`, `reuters`, or `general`. Other names use the `general` parser while still naming the output folder. For example, `--corpus webtext` selects `general`.

### Split an existing database

The split is by source document, so sentences from one document stay together. Ratios are in `train test` order for two values, or `train valid test` order for three. They must be positive and sum to 1. The seed makes document assignment reproducible.

```bash
python src/data-pipeline/data_pipeline.py \
  --corpus webtext \
  --in data/processed/webtext/webtext.db \
  --out data/processed \
  --split-ratios 0.8 0.2 --seed 42
```

For a split-only run, `--in` can be the database path or a folder containing `<corpus>.db`. If `--in` is omitted, the pipeline looks for `data/processed/<corpus>/<corpus>.db`. The resulting sentence files are `<out>/<corpus>/train.txt` and `test.txt`. For three ratios, use for example `--split-ratios 0.8 0.1 0.1`; this also writes `valid.txt`, the validation filename expected by the model data loader.

If `--load` and a split are both selected, the split uses the database created during that run. A fresh load removes prior shelve backing files and creates a matching `.db.sha` checksum. A split-only run verifies the checksum before reading the cached DB.

### Check rare n-grams

```bash
python src/data-pipeline/data_pipeline.py \
  --corpus webtext \
  --in data/processed/webtext \
  --out data/processed \
  --rare-check
```

Without a split, the pipeline reads `train.txt` and `test.txt` from `--in` (or from `data/processed/<corpus>/` if `--in` is omitted). It also recognizes `train_sentences.txt` and `test_sentences.txt`. For a three-way split, rare n-gram analysis uses train for vocabulary/counts and combines valid plus test as one held-out chunk.

The report is written to `<out>/<corpus>/rare-ngrams.csv`. Candidate discovery samples up to 20,000 test sentences; counts for discovered candidates are then counted over the full train and test files. This CSV is an example list, not a complete inventory or corpus-wide unseen-ngram percentage.

## Config-driven preparation

Run `data_orchestrator.py` with a model YAML containing a `data:` section. `data.yaml` in this folder remains the source of truth for each corpus download/source location.

The orchestrator reads the proposed keys below. For existing model configs, it also accepts `data.loader` as the corpus name and `data.params.ratios` / `data.params.split_seed` as fallbacks.

```yaml
data:
  corpus: brown
  split_seed: 42
  split_ratio: [0.8, 0.1, 0.1] # train, valid, test; two values make train/test
  use_cached: true
  input: data/raw               # root folder; source subfolder comes from data.yaml
  output: data/processed
  rare-check: true
  clean: true
```

```bash
python src/data-pipeline/data_orchestrator.py --config src/configs/brown.yaml
```

The orchestrator uses a cached DB only when its `.db.sha` file is present and matches. Otherwise, it resolves the raw corpus from the manifest; it extracts an existing NLTK ZIP as needed. If no source files are available locally, it downloads through the manifest, then runs clean (when requested), load, split, and rare-check. A cached database and its checksum can be shared without the raw corpus.

Logs show short stage names and project-relative output paths by default. Per-document load details and split sample sentences are hidden; add `--verbose` to the orchestrator or pipeline command to show them.

## Input format and corpus notes

- Cleaning and loading expect `.txt` input files. Each parsed document becomes a database record; sentence splitting is handled by the selected parser.
- `--corpus` is a simple folder name, such as `gutenberg` or `webtext`; it cannot contain a path.
- WordNet and OMW are lexical databases (`.noun`, `.tab`, and related files), not sentence corpora. Convert selected definitions/examples to text first if you intend to experiment with them; keep that derived data separate from ordinary prose corpora.
- The scripts require NLTK, `datasketch`, `langdetect`, and Beautiful Soup in the active Python environment. `charset-normalizer` is optional; automatic encoding mode falls back to CP1252 when it is unavailable.
