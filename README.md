# NLP Language-Model Experiments

Word-level statistical and neural language models for the NLP assignment. The
document-budget suite compares independently fitted models at cumulative training
budgets of 10%, 20%, ..., 100%, using a vocabulary built only from each budget's
training documents.

## Start Here

| Area | Guide |
| --- | --- |
| Google Colab pilot, private bundle and result retrieval | [Colab guide](colab/README.md) |
| Local training, interactive CLI and saved artifacts | [Source guide](src/README.md) |
| Model package, scoring contract and run lifecycle | [Model guide](src/lm/README.md) |
| Configuration schemas and new experiment versions | [Configuration guide](src/configs/README.md) |
| Synthetic tests and their limits | [Test guide](src/tests/README.md) |
| Corpus preparation and legacy download pipeline | [Pipeline guide](src/data-pipeline/README.md) |
| Required local corpus files and provenance | [Prepared-data guide](data/processed/readme.txt) |
| Scientific protocol and remaining research work | [Experiment plan](docs/research_directions/data_efficiency/plan.md) |

Run the commands below from the repository root. Python 3.12 is recommended for
local execution; the completed Colab pilot used Python 3.13 and a Tesla T4.

```sh
uv venv ~/.venvs/nlp-lm --python 3.12
uv pip install --python ~/.venvs/nlp-lm/bin/python -r src/requirements.txt
PYTHONPATH=src ~/.venvs/nlp-lm/bin/python -B -m unittest discover -s src/tests -v
~/.venvs/nlp-lm/bin/python src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage all --dry-run
```

For a minimal Linux CPU install, install PyTorch from its CPU-only wheel index,
then install the remaining project dependencies from `req.txt`. This avoids pip
choosing the default CUDA-enabled PyTorch wheel from PyPI. Notebook and optional
Hugging Face packages are omitted:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r req.txt
```

Use Python 3.10 or newer.
The Python build must include the `dbm.gnu`/gdbm standard-library backend for
the cached `shelve` corpus database; this is provided by the OS/Python package,
not by pip.

Keep the virtual environment outside cloud-synced folders. With plain `pip`,
create the environment using `python3.12 -m venv ~/.venvs/nlp-lm`, then run
`~/.venvs/nlp-lm/bin/python -m pip install -r src/requirements.txt`.

## Inputs, Processing And Outputs

Commit `f078584` adds the audited document-budget experiment suite and Colab
workflow. Commit `d2483da` preserves legacy checkpoint loading after that
integration. The existing positional-config runner and interactive CLI remain
available; the workflow below is selected explicitly with `--suite`.

### Required Inputs

For local preparation and training, restore these inputs before running a real
experiment. Paths below are relative to the repository root.

| Input | Purpose |
| --- | --- |
| [Pilot configuration](src/configs/data_efficiency_nltk_pilot.yaml) | Selects corpora, source paths, budgets, seeds, vocabulary rules, models, training settings, resource limits and `output_root` |
| `data/processed/<corpus>/train.txt` and `test.txt` | Original imported exports, one nonempty sentence per line; test membership must remain unchanged |
| `data/processed/<corpus>/<corpus>.db` | Original SQLite document store, used to recover whole source documents and reconcile them with the exports |
| Matching `<corpus>.db.sha` sidecar | Integrity metadata for the document store |
| [Import manifest](data/processed/corpus_import_manifest.json) | Records imported file provenance and checksums |
| Python environment from [requirements](src/requirements.txt) | Runs preparation, statistical/neural models, evaluation and reporting |

The pilot selects `gutenberg`, `webtext` and `inaugural`. It does not need a
pre-existing `valid.txt`: validation is derived from the imported training
documents. Historical `rare-ngrams.csv` files are not suite inputs.

For **Colab**, upload the private bundle produced by `lm.colab`, not the original
corpus folders. The bundle contains code, configuration, prepared manifests and
SQLite sentence indexes. The [notebook](colab/nltk_data_efficiency.ipynb) also
needs its matching `EXPECTED_BUNDLE_SHA256`, a suitable runtime and Drive access
for durable outputs. See the [bundle instructions](colab/README.md#build-a-matching-bundle).

### What It Does

1. Verifies source checksums, reconciles exported sentences to whole source
	documents and checks split overlap/duplicates. It does not download corpora or
	rebuild the original imported databases.
2. Applies the configured WebText repair in a separate derived directory:
	preserves test bytes and removes overlapping training source groups.
3. Holds out validation from the existing training documents using data seed
	`42`. Test remains the imported test population. It orders the remaining
	training pool using budget seed `55` and selects cumulative document budgets
	of 10%, 20%, ..., 100%, rounding document counts up.
4. Builds a fresh vocabulary from each budget's training sentences only:
	lowercase, whitespace tokenization, `min_freq=2`, no vocabulary cap. All
	models at that corpus/budget share the same token IDs; held-out words do not
	enter vocabulary construction.
5. Fits bigram MLE, trigram MLE, validation-selected trigram add-k, LSTM and
	context-2 FFNN independently at each budget. Neural fits use three epochs
	and final seed `42`; validation selects settings/best weights.
6. Seals the required training artifacts and their hashes before test scoring,
	then evaluates test data and generates tables, plots and an offline report.

The default pilot produces 150 final fits and 90 validation-only add-k tuning
fits. Whole-document rounding can repeat budgets, especially for WebText; aliases
are recorded rather than presented as different training populations.

### Run Locally

All commands run from the repository root. The dry run in [Start Here](#start-here)
needs no corpus data and only displays the planned experiment matrix. After
restoring the private inputs, run the stages separately to inspect progress:

```sh
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage prepare
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage train --resume
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage seal
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage evaluate
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage report
```

`--stage all` runs the full sequence. `--resume` verifies and reuses completed
fits; an interrupted fit restarts rather than resuming optimizer state mid-epoch.
For changed code, inputs or settings, copy the configuration and choose a new
`name` and `output_root` before starting. Do not overwrite the completed pilot.

### Legacy Model Runs And Tokenizer Diagnostics

The original corpus-configured runner remains available for individual neural
experiments. It supports epoch-boundary checkpoint resume and tokenizer selection
(`space`, `wp` for WordPunct, or `tb` for Treebank):

```sh
python src/run.py src/configs/brown.yaml --algorithm ffnn --context-size 1 --token wp
python src/run.py src/configs/brown.yaml --algorithm ffnn --context-size 1 --token wp --resume
```

For a full corpus sweep, use the mega-run manifest. It lists corpus experiment
configs and the training percentages to run:

```yaml
run:
  - inaugural.yaml
train_percentages: [20, 40, 60, 80, 100]
```

Run it from the repository root:

```sh
python src/run.py --full src/configs/mega-run.yaml
python src/run.py --full src/configs/mega-run.yaml --parallelism 2
```

For a smaller end-to-end check, `mega-smoke.yaml` uses Inaugural at 20% and 40%:

```sh
python src/run.py --full mega-smoke.yaml
```

For each corpus, the full runner prepares or reuses the configured corpus data,
trains every configured model and seed at each percentage, runs
`src/model_comparison.py`, and checks for the expected model and report files
before advancing. `--parallelism` runs separate corpus pipelines concurrently;
it defaults to `1`.

Models and the exact training snapshots are stored under
`results/<config name>/train_<percent>pct/`. Comparison artifacts are grouped
with them under `results/<config name>/metrics/`: aggregate `metrics.csv`,
prediction CSVs, `charts.png` for trends across percentages, and one
`plot_<percent>p.png` per percentage. Each fixed-percentage plot compares
accuracy and perplexity across models and train, validation, test, and rare
examples. See the [source guide](src/README.md#full-corpus-run) for details.

To compare tokenizer token/type counts before a run, use:

```sh
python src/analyze_tokenizer_vocab.py data/processed/brown/train.txt
```

See the [LM guide](src/lm/README.md#shared-tokenization) for tokenizer and resume
details, and the [pipeline guide](src/data-pipeline/README.md) for reproducible
document sampling and corpus preparation options.

### Where Outputs Go

| Execution | Output location |
| --- | --- |
| Local preparation/training | Configured `output_root`; default `results/nltk_data_efficiency_pilot_v2/` |
| Local bundle creation | `colab/nltk_pilot_bundle.zip` and its `.zip.sha256` sidecar |
| Colab with Drive enabled | `MyDrive/NLP_Experiments/<OUTPUT_NAME>/`; default name `nltk_data_efficiency_pilot_v2` |
| Downloaded completed pilot on this machine | `results/colab_imports/20261026T175208Z/nltk_data_efficiency_pilot_v2/` |

Inside a suite output directory, start with these artifacts:

| Relative path | Content |
| --- | --- |
| `report.html` | Offline experiment report; keep its relative assets beside it |
| `runs.csv`, `metrics.csv`, `summary.csv`, `diagnostics.csv` | Per-run metrics, summaries and diagnostic results |
| `protocol.json`, `seal.json`, `environments/`, `events.jsonl` | Experiment identity, artifact integrity, environment and execution audit |
| `prepared/<corpus>/manifest.json` | Source hashes, exact validation/test document IDs and each budget's training document IDs |
| `prepared/<corpus>/sentences.sqlite` | Prepared sentence content for local runs; Colab preserves the indexes in `input_bundle.zip` instead of necessarily copying them here |
| `curated/webtext/` | Derived repair audit and repaired inputs; original corpus files are untouched |
| `<corpus>/budget_010/vocab.json`, `vocab_counts.csv`, `data_stats.json` | Ordered vocabulary, training-token counts, selected document IDs and verification hashes for the 10% budget |
| `<corpus>/budget_010/final/<model>/seed<seed>/` | Run config, validation/test metrics, fitted model, status, and neural epoch histories where applicable |
| `<corpus>/budget_010/tuning/` | Validation-only candidate fits and their artifacts |

Other budgets use `budget_020` through `budget_100`. Neural weights are `.pt`
files; count-model state is `counts.pkl`. Load only trusted checkpoints/pickles.
Histories record training progress, not separate final training-set rescoring.
See the [complete artifact layout](src/README.md#saved-artifacts).

### Recover Splits And Vocabulary

A run's `config.yaml` contains `suite_run.prepared`, `suite_run.subset` and
`suite_run.vocab` fingerprints. Match them to the corpus's prepared manifest and
the budget's `data_stats.json`; hashes identify content, not a way to regenerate
missing text. The prepared SQLite index maps document IDs to original
`source_key` names and sentence content. Preserve it or the original imported
corpora in private storage, together with the manifests/input bundle.

To recreate just the vocabulary, the budget's ordered `vocab_counts.csv` is
sufficient. Run this from the repository root after installing dependencies;
change `budget` to the desired local or downloaded corpus/budget directory:

```python
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from lm.data import Vocab

budget = Path("results/nltk_data_efficiency_pilot_v2/gutenberg/budget_010")
with (budget / "vocab_counts.csv").open(encoding="utf-8", newline="") as stream:
	 counts = Counter({
		  row["word"]: int(row["training_count"])
		  for row in csv.DictReader(stream)
	 })
vocab = Vocab.from_counts(counts, min_freq=2)
stats = json.loads((budget / "data_stats.json").read_text())
assert vocab.fingerprint == stats["vocab_hash"]
```

Use the frequency cutoff from that experiment's configuration. Preserve CSV row
order: equal-frequency tokens use first-occurrence order for their IDs. The
first four IDs are `<pad>`, `<unk>`, `<s>` and `</s>`; `vocab.itos` is the ordered
list and `vocab.stoi` is the reverse mapping. Reconstruction was verified against
the saved token IDs and hashes for all 30 completed pilot budgets. This does not
recover sentence order or the corpus itself from token counts.

## Colab Experiment

The [notebook](colab/nltk_data_efficiency.ipynb) uses a locally generated private
bundle containing source code and audited prepared sentence indexes. It does not
clone this repository, redownload corpora, or regenerate imported test splits.
Follow the [Colab guide](colab/README.md) to build the bundle and update the
notebook's checksum before uploading it.

The completed pilot covers Gutenberg, WebText and Inaugural, five models, ten
budgets, three neural epochs and one final neural seed: 150 final fits plus 90
validation-only add-k tuning fits. Models are reported in this order:
`bigram_mle`, `trigram_mle`, `trigram_addk`, `lstm`, `ffnn`.

This is a pilot, not convergence or multi-seed research evidence. WebText has only
two independent training source groups after the approved overlap repair and
validation split, so several rounded budgets repeat. Perplexity across budgets
also reflects vocabulary growth and changing prediction support. Infinite MLE
scores caused by unseen targets are retained, not replaced with finite values.

## What Git Contains

Code, notebook source, configuration, tests, documentation and the corpus import
manifest are versioned. Corpus contents, ZIP bundles, databases, model checkpoints
and root-level generated results are excluded by [.gitignore](.gitignore).
Historical tracked results under `src/results/` are separate from the new suite.

A fresh clone can run the synthetic tests and dry-run command without corpus
downloads. Real preparation/training requires the original private imported
corpora and their matching provenance files, or a separately supplied verified
Colab bundle. Do not fabricate missing files, disable checksum checks, or
regenerate held-out splits to make a run proceed.

## Results And Reproducibility

The verified pilot produced complete validation/test metrics, vocabularies, token
counts, configurations, checkpoints, three-epoch neural histories, plots, logs,
partition manifests and the preserved input bundle. Training histories are not
separate final training-set evaluations. See the [batch report](docs/research_directions/data_efficiency/batch-report.yaml)
for the completed pilot and outstanding broader tasks.

Keep full result archives on private storage. A run's `report.html` works offline
when its relative figure/artifact files are kept beside it. Configuration or
implementation changes require a new output directory; do not overwrite or
resume the completed pilot with changed inputs.

Check each corpus's distribution terms before sharing data or private bundles.
Availability through NLTK does not imply a common licence for all corpora.
