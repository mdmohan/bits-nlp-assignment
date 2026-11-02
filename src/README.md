# Neural Language Models (PS-1)

A small, generic PyTorch framework for training and comparing next-word language models.
Any dataset and any model can be plugged in; the first experiment compares a
**feed-forward neural LM** and an **LSTM LM** on the NLTK Brown corpus.

## Project layout

```
src/
├── run.py                 # command-line entry point
├── requirements.txt
├── configs/
│   └── brown.yaml         # Brown press genres: FFNN vs LSTM, 3 seeds
├── lm/
│   ├── data.py            # pipeline split reader, Corpus, Vocab, batching
│   ├── controller.py      # runs models x seeds on one dataset and saves results
│   ├── neural/
│   │   ├── __init__.py    # neural model registry (MODELS, build_model)
│   │   ├── trainer.py     # training loop, evaluation, early stopping, seeding
│   │   ├── ffnn/          # FeedForwardLM
│   │   └── lstm/          # LSTMLM
│   └── statistical/       # count-based models (n-gram etc.)
└── results/<experiment>/  # created by run.py
```

## Setup

Python 3.12 is recommended. Run setup from the repository root:

```sh
# with uv
uv venv ~/.venvs/nlp-lm --python 3.12
uv pip install --python ~/.venvs/nlp-lm/bin/python -r src/requirements.txt

# or with plain pip
python3.12 -m venv ~/.venvs/nlp-lm
~/.venvs/nlp-lm/bin/python -m pip install -r src/requirements.txt
```

Keep the virtual environment **outside OneDrive / cloud-synced folders**; on-demand file
eviction makes imports hang.

Dataset download, cleaning, loading, and splitting run through the corpus pipeline
configured in `data.corpus`. See `src/data-pipeline/README.md` for the manifest and
pipeline behavior. The model loader consumes the resulting `train.txt`, `valid.txt`,
and `test.txt` files.

## Running

From the `src` directory:

```sh
~/.venvs/nlp-lm/bin/python -u run.py configs/brown.yaml
```

This trains every model listed in the config for every seed, prints per-epoch progress, and
writes results to `results/<name>/train_<percent>pct/` (for example,
`results/brown_press_neural/train_100pct/`). Each training percentage has its own
folder, so repeated runs do not overwrite each other's checkpoints or metrics.

Run one configured algorithm with `--algorithm` (or its alias `--model`). Override the configured
seed list with `--seed`; when used alone, the seed override still runs every configured algorithm:

```sh
python -u run.py configs/brown.yaml --algorithm ffnn --seed 13
python -u run.py configs/brown.yaml --model lstm
```

With no overrides, the configured model and seed lists are used. The saved `config.yaml` records
the effective model and seed selection for that output.

The FFNN's `context_size` is the number of preceding words it receives to predict the next
word. Override it for a controlled context comparison while keeping the same corpus, seed,
and training percentage:

```sh
python -u run.py configs/brown.yaml --algorithm ffnn --context-size 1 --seed 13
python -u run.py configs/brown.yaml --algorithm ffnn --context-size 2 --seed 13
python -u run.py configs/brown.yaml --algorithm ffnn --context-size 4 --seed 13
```

Context size `1` gives a bigram-style neural context; `2` gives a trigram-style context.
Each override writes under `results/<name>/train_<percent>pct/ffnn_ctx<N>/`. Within the
experiment, every configured model variant has its own `models/<variant>/` folder, and its
context size is recorded in `runs.csv`, `summary.csv`, and the saved config. The option applies
to FFNN only.

To run several FFNN context sizes in one experiment, add a top-level list beside `seeds` in
the YAML config:

```yaml
seeds: [13, 42, 2026]
context_sizes: [1, 2, 4]
```

Each configured FFNN is trained once per context size and seed; other configured models run
once per seed. Each variant gets a separate folder under `results/<name>/train_<percent>pct/models/`, preventing
duplicate algorithm entries from overwriting checkpoints or histories. The run table reports
the variant and context size. `--context-size N` still overrides the list for a one-context run.

For a quicker run, pass `--train-percent` to use only the first portion of the training split;
validation and test data remain complete:

```sh
python -u run.py configs/brown.yaml --train-percent 10
```

The default is `100`, which uses all training sentences. The chosen percentage is recorded in
the output config, data statistics, and run table. Each percentage is saved in its matching
`train_<percent>pct/` results subfolder.

### Full corpus run

Use a mega-run manifest to prepare each corpus once, train every configured model and seed at
each listed percentage, compare the saved runs, and verify the expected artifacts before
moving to the next corpus. The default manifest is `configs/mega-run.yaml`:

```sh
python -u run.py --full configs/mega-run.yaml
python -u run.py --full configs/mega-run.yaml --parallelism 2
```

The manifest's `run` list names corpus configs and `train_percentages` selects the training
fractions. If `train_percentages` is omitted, the runner uses 20, 40, 60, 80, and 100 percent.
`--parallelism` defaults to `1`; higher values run independent corpus pipelines in separate
processes.

Each run keeps model artifacts and its exact training snapshot under
`results/<config name>/train_<percent>pct/`. The comparison writes to
`results/<config name>/metrics/`: `metrics.csv`, `perplexity_by_percentage.csv`, prediction
CSVs, `charts.png` for trends by percentage, and a fixed-percentage `plot_<percent>p.png` for
each run. Fixed-percentage plots show accuracy and perplexity across models for train, valid,
test, and rare examples.

### Inspecting a saved run

Run the CLI examples in this section from the repository root, not from `src`.

Start the interactive CLI without loading anything first:

```sh
python src/lm/cli.py
```

Inside the CLI, create and evaluate unsmoothed or smoothed n-gram models:

```text
lm> ngram mygutenberg --n 2 --train data/processed/gutenberg/train.txt
lm> ngram mygutenberg --evaluate data/processed/gutenberg/test.txt
lm> ngram mygutenberg_laplace --n 2 --train data/processed/gutenberg/train.txt --smoothing laplace --k 1.0
lm> ngram mygutenberg_backoff --n 3 --train data/processed/gutenberg/train.txt --smoothing backoff --discount 0.75
```

Laplace requires an explicit `--k` value. Backoff accepts `--discount` (default `0.75`) and
backs off recursively: trigram to bigram to unigram. The trigram build collects those lower-order
counts from the same training file, so no separate bigram model build is needed.

Evaluation reports perplexity and top-1 accuracy over the full file, then prints up to 25
correct and 25 incorrect n-gram predictions separately. Examples are ranked by the model's
probability for its top prediction, so incorrect examples show the most confident mistakes.
Each row includes `P(expected)` and `P(prediction)`. Set `--limit N` to show up to N examples
per section; `--limit 0` prints every event.

To compare against a neural checkpoint, load it and compare both models on the same test file:

```text
lm> nn load results/brown_press_neural/train_100pct/models/ffnn_ctx1/ffnn_seed13.pt
lm> vocab 10
lm> predict the cat
lm> ngram brown_bigram --n 2 --train results/brown_press_neural/train_100pct/train.txt
lm> compare --one ngram brown_bigram --two neural ffnn_seed13.pt --test data/processed/brown/test.txt
lm> quit
```

When the n-gram is created from the neural model's saved `train.txt`, the CLI uses the
checkpoint's saved vocabulary and lowercasing rule, and reads the snapshot in full. Otherwise
it builds a vocabulary from the supplied training text (`--min-freq 2` by default). N-gram models are
kept in memory for the current CLI session; recreate them after restarting the CLI. A zero
probability event gives an unsmoothed model infinite perplexity.

Loading requires the checkpoint, `config.yaml`, and `vocab.json` in the same run directory.
The historical Brown weights under `source_code/results/` and metadata under `src/results/`
are not currently a directly loadable bundle. The checkpoint path above is an example of
the required co-located layout, not a claim that those historical files have been moved.

### Using it from Python / a notebook

```python
from lm.data import load_corpus_files
from lm.controller import LanguageModelController
from lm.neural.trainer import TrainConfig

corpus = load_corpus_files(
    "data/processed/brown/train.txt",
    "data/processed/brown/valid.txt",
    "data/processed/brown/test.txt",
)
controller = LanguageModelController(corpus, min_freq=2, batch_size=32, output_dir="results/notebook")
result = controller.run("lstm", {"hidden_dim": 256}, TrainConfig(epochs=10), seed=42)
print(result["test_ppl"])
```

## Configuration

Each experiment is one YAML file. `configs/brown.yaml`:

| Key | Meaning |
|---|---|
| `name` | Experiment name; output folder is `<output_dir>/<name>` |
| `output_dir` | Root folder for results (default `results`) |
| `device` | `auto` (CUDA, then Apple MPS, then CPU), or `cpu` / `cuda` / `mps` |
| `seeds` | One full training run per model per seed |
| `data.corpus` | Corpus key from `src/data-pipeline/data.yaml`; selects pipeline inputs and outputs |
| `data.split_ratios` | Two ratios for train/test or three for train/valid/test; neural training needs three |
| `data.split_seed` | Reproducible document-level split seed |
| `data.input` | Raw-data root; corpus-specific location is resolved from the manifest |
| `data.output` | Processed-data root; pipeline writes the split files under the corpus name |
| `data.use_cached` | Reuse a database only when its checksum sidecar validates |
| `data.clean`, `data.rare_check` | Enable optional cleaning and rare-sequence analysis |
| `data.lowercase` | Lowercase tokens after reading the pipeline split files (default `true`) |
| `vocab.min_freq` | Training words seen fewer times become `<unk>` |
| `vocab.max_size` | Optional cap on vocabulary size |
| `models` | List of `{name, params}`; `name` is a key of `MODELS` in `lm/neural/__init__.py` |
| `training` | `batch_size`, `epochs`, `lr`, `grad_clip`, `weight_decay`, `patience` |

### Available datasets

Corpus sources are listed in `src/data-pipeline/data.yaml`. Add or select a manifest
entry there; the data orchestrator resolves local raw files or downloads them before
the neural model reads its three split files.

### Available models

| Name | Class | Description | Params |
|---|---|---|---|
| `ffnn` | `FeedForwardLM` | Bengio et al. (2003) style: embeddings of the previous `context_size` words are concatenated and passed through one tanh hidden layer | `context_size`, `embed_dim`, `hidden_dim`, `dropout` |
| `lstm` | `LSTMLM` | Word-level LSTM using the whole sentence prefix | `embed_dim`, `hidden_dim`, `num_layers`, `dropout` |

## Extending

**New dataset:** add its source and download metadata to the data-pipeline manifest,
then use the orchestrator to create sentence-level split files. `lm/data.py` stays
focused on reading those files and constructing the vocabulary and batches used by models.

**New neural model:** add a subpackage under `lm/neural/` (like `ffnn/`) with an `nn.Module`
that follows this contract, and add it to `MODELS` in `lm/neural/__init__.py`:

```python
model = MyModel(vocab_size, pad_id, **params)
logits = model(inputs)  # (batch, seq_len) word ids -> (batch, seq_len, vocab_size)
```

`logits[:, t]` must predict the word after `inputs[:, t]` using only `inputs[:, :t+1]`.
The trainer and controller need no changes.

## Method details

- **Splitting:** documents are shuffled with `split_seed` and split 80/10/10, so no document
  appears in more than one split. Validation is used only for early stopping.
- **Vocabulary:** built from the training split only. Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`.
- **Sequences:** each sentence is modelled independently as `<s> w1 ... wn </s>`;
  the input is `<s> w1 ... wn` and the target is `w1 ... wn </s>`.
- **Perplexity:** `exp(total negative log-likelihood / number of predicted tokens)` over the
  whole split. Predicted tokens are the words plus one `</s>` per sentence; `<s>` and padding are
  not counted. Sentence perplexities are never averaged.
- **OOV rate:** fraction of word tokens in a split that are not in the vocabulary
  (they are scored as `<unk>`).
- **Training:** Adam, gradient clipping, early stopping on validation perplexity with `patience`;
  the best-validation weights are restored before testing.
- **Reproducibility:** `random`, `numpy` and `torch` are seeded before each run. Runs on Apple
  MPS / CUDA may still show tiny non-deterministic differences.

## Outputs

`results/<name>/train_<percent>pct/` contains aggregate run files and a `models/` folder with
one subfolder per configured model variant. Each model folder contains its checkpoint(s),
training history, config, vocabulary, and data statistics. The exact selected training
sentences are copied to `train.txt` in that percentage folder for matched n-gram comparisons.
For example, the 20% and 40% runs are kept separately under `train_20pct/` and `train_40pct/`.

| File | Content |
|---|---|
| `runs.csv` | One row per model variant x seed: parameters, vocab size, test OOV rate, valid/test perplexity, epochs, train and test time |
| `summary.csv` | Mean and standard deviation per model variant across seeds |
| `train.txt` | Exact nonempty sentence lines consumed by the neural run at this training percentage |
| `models/<variant>/<model>_seed<seed>_history.csv` | Per-epoch train/valid perplexity and time |
| `models/<variant>/<model>_seed<seed>.pt` | Best model weights (not committed to git; regenerate with `run.py`) |
| `vocab.json` | Vocabulary list (index = word id) |
| `data_stats.json` | Sentence/token counts per split, vocab size, OOV rates |
| `config.yaml` | Copy of the config used |

Full-run comparison files are stored separately in `results/<name>/metrics/`. The aggregate
metrics retain train, valid, test, and rare scores; prediction-detail CSVs contain held-out
valid/test and rare examples.

## Results: Brown press genres

Data: Brown `news`, `editorial`, `reviews` (88 documents), lowercased, `min_freq=2`.

| Split | Sentences | Tokens |
|---|---|---|
| Train | 7,475 | 161,092 |
| Valid | 806 | 18,561 |
| Test | 1,090 | 23,209 |

Vocabulary 8,624 (incl. specials); test OOV rate 12.1%; 24,299 predicted test tokens.

| Model | Parameters | Epochs | Valid PPL | Test PPL (mean ± std, 3 seeds) | Train time (s) | Test time (s) |
|---|---|---|---|---|---|---|
| FFNN (3-word context) | 3.42 M | 8-9 | 196.1 ± 0.6 | **173.7 ± 0.4** | 55.5 ± 5.5 | 0.20 |
| LSTM (1 layer) | 3.72 M | 12 | 170.7 ± 0.6 | **150.4 ± 0.5** | 87.5 ± 1.1 | 0.39 |

Seeds 13, 42, 2026. Hardware: Apple Silicon Mac, PyTorch MPS backend. The full run takes about
8 minutes.

Environment used: Python 3.12, torch 2.14.1, nltk 3.10.3, pandas 3.0.6, pyyaml 6.0.3, datasets 5.1.0.

## Data licence

The Brown Corpus is distributed through NLTK data for non-commercial use; it is not public domain.
See <https://www.nltk.org/nltk_data/>.

## Document-Budget Sweep And Colab

See the [Colab guide](../colab/README.md) for bundle/checksum pairing, Drive
authorization, resume and complete downloads; the [model guide](lm/README.md)
documents scoring and artifact trust boundaries. Configuration schemas and
synthetic coverage are described in [configs/README.md](configs/README.md) and
[tests/README.md](tests/README.md).

The opt-in suite reuses `CountNGram`, `NgramLM`, the registered neural models and
`Trainer`. Interactive comparison and the suite share a structured evaluator.
Legacy positional-config commands remain available; their `--train-percent` still
selects sentence prefixes, not document budgets.

The [pilot config](configs/data_efficiency_nltk_pilot.yaml) selects only Gutenberg,
WebText and Inaugural: ten cumulative document budgets (10% through 100%), three
epochs, seed 42, bigram/trigram MLE, validation-selected trigram add-k, LSTM and
context-2 FFNN. This is a short pilot, not a convergence claim. Later research runs
need additional seeds and validation search, with a new output root.

Run from the repository root:

```sh
~/.venvs/nlp-lm/bin/python src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage all --dry-run
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage prepare
PYTHONPATH=src ~/.venvs/nlp-lm/bin/python -u -m lm.colab --suite src/configs/data_efficiency_nltk_pilot.yaml --output colab/nltk_pilot_bundle.zip
```

Upload [the notebook](../colab/nltk_data_efficiency.ipynb) to Google Colab, select
a T4 GPU, then run its cells in order. Its upload cell accepts the checksum-pinned
private bundle; authorize Drive directly in the browser. Results are written beneath
`MyDrive/NLP_Experiments/nltk_data_efficiency_pilot_v2`, with a full ZIP export at the
end. Rebuilding the bundle changes its ZIP checksum; update the notebook's
`EXPECTED_BUNDLE_SHA256` to the generated sidecar value before uploading the rebuilt
pair. Neither the notebook nor suite redownloads or rebuilds imported corpora.
Prepared corpora, bundle ZIPs, checkpoints and root-level results are not in Git.
A fresh clone needs the original private inputs before preparation, or a separately
supplied verified bundle for Colab. The checked-in notebook pins the archived
pilot bundle; rebuilding edited code requires its new hash and a new output name.

For staged local execution, use `--stage train --resume`, then `seal`, `evaluate`,
and `report`. `train` never evaluates test metrics. `seal` checks required fits,
settings, partitions and vocabularies; `evaluate` rejects changes. `all` runs these
stages in order. `--corpora inaugural` restricts execution. Code/config changes need
a new output root. Completed fits resume only when environment fingerprints and
artifact hashes match. Interrupted fits restart; previous attempt logs and latest
epoch snapshots remain available.

### Saved Artifacts

```text
results/<suite>/
  protocol.json, environments/, events.jsonl, corpus_status.json, seal.json
  prepared/<corpus>/manifest.json, sentences.sqlite
  curated/webtext/repair.json, grouped.db, train.txt, test.txt
  <corpus>/budget_<percent>/
    vocab.json, vocab_counts.csv, data_stats.json, *_selection.json
    tuning/<model>/<candidate>/...
    final/<model>/seed<seed>/
      config.yaml, vocab.json, data_stats.json, result.json, status.json
      history.csv, training.png, report.md
      validation_metrics.json, test_metrics.json, evaluation_status.json
      <model>_seed<seed>.pt or counts.pkl
      attempts/<timestamp>/run.log, history.csv, latest.pt
  <corpus>/figures/*.png
  runs.csv, metrics.csv, summary.csv, diagnostics.csv, examples.json, report.html
```

Reports include per-run epoch curves and per-corpus perplexity, accuracy,
vocabulary/OOV, parameter/count size, runtime, process memory high-water and zero-event
charts. Metrics include frequency/novelty strata and reproducible full-prefix cases.
Tables and CSV exports use `bigram_mle`, `trigram_mle`, `trigram_addk`, `lstm`,
`ffnn` within each budget. Training histories/fitting statistics are not separate
final training-set evaluation metrics. Colab retains prepared indexes inside the
preserved `input_bundle.zip`; its output `prepared/` directories contain manifests,
not necessarily another copy of each sentence database.
Infinite MLE scores are JSON `null` plus an explicit status and zero-event count,
not clipped finite values or silently omitted seeds. Count checkpoints are Python
pickles: load only trusted run bundles.

Each budget's vocabulary uses only that budget's training documents, with the same
frequency cutoff and shared IDs across all models. Prediction support excludes
PAD/BOS and includes UNK/EOS. One nonempty prepared line is one sentence, contributing
word targets plus EOS. Cross-budget perplexity reflects both data and vocabulary
growth; inspect actual word counts, vocabulary and OOV too.

WebText's approved derived repair keeps test bytes unchanged and removes all training
variants of test source groups (`overheard.txt` and `singles.txt`). Remaining variants
stay grouped by source; original within-group weighting is retained. After validation
only two training groups remain. Rounded aliases are recorded as `alias_of_percent`,
not ten distinct training populations. Gutenberg's book-sized units also make
percentages approximate. Original imports and the download registry are unchanged.
Historical rare-ngrams CSVs use a different tokenizer and are not relabelled as suite
diagnostics.

```sh
PYTHONPATH=src ~/.venvs/nlp-lm/bin/python -B -m unittest discover -s src/tests -v
```

The seven-corpus rollout and research acceptance gates remain separate from this
pilot. See the [implementation plan](../docs/research_directions/data_efficiency/plan.md).
