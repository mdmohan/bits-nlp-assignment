# Neural Language Models (PS-1)

A small, generic PyTorch framework for training and comparing next-word language models.
Any dataset and any model can be plugged in; the first experiment compares a
**feed-forward neural LM** and an **LSTM LM** on the NLTK Brown corpus.

## Project layout

```
source_code/
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

Python 3.12 is recommended.

```sh
# with uv
uv venv ~/.venvs/nlp-lm --python 3.12
uv pip install --python ~/.venvs/nlp-lm/bin/python -r requirements.txt

# or with plain pip
python3.12 -m venv ~/.venvs/nlp-lm
~/.venvs/nlp-lm/bin/pip install -r requirements.txt
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
writes results to `results/<name>/`.

Run one configured algorithm with `--algorithm` (or its alias `--model`). Override the configured
seed list with `--seed`; when used alone, the seed override still runs every configured algorithm:

```sh
python -u run.py configs/brown.yaml --algorithm ffnn --seed 13
python -u run.py configs/brown.yaml --model lstm
```

With no overrides, the configured model and seed lists are used. The saved `config.yaml` records
the effective model and seed selection for that output.

For a quicker run, pass `--train-percent` to use only the first portion of the training split;
validation and test data remain complete:

```sh
python -u run.py configs/brown.yaml --train-percent 10
```

The default is `100`, which uses all training sentences. The chosen percentage is recorded in
the output config, data statistics, and run table.

### Inspecting a saved run

From the repository root, load a result folder and select one of its saved checkpoints:

```sh
python src/lm/cli.py load src/results/brown_press_neural --checkpoint ffnn_seed13.pt
```

Replace `src/results/brown_press_neural` with the result folder created by your training run.

Inside the prompt, use `vocab [N]` for the most frequent vocabulary entries, `stats` for
split/OOV information, or enter text (or `predict TEXT`) to see the loaded neural model's
next-word guesses. `evaluate FILE bi|tri` evaluates both an unsmoothed count n-gram model and
the loaded neural model on that context window. It reads counts from the configured training
split; pass `--train PATH` to `load` if that split has moved. `rare-check bi|tri CSV [RANGE]` scores
the selected model on matching CSV rows, weighted by `test_count`; the range is zero-based and
end-exclusive, defaults to `0-10`, and accepts `0-all` to use every matching row. An unsmoothed n-gram has
infinite perplexity if any evaluated event has zero probability; the CLI also reports its
finite-event perplexity and coverage.
```text
lm> vocab 10
lm> stats
lm> The cat sat
lm> evaluate data/processed/brown/test.txt tri
lm> rare-check bi data/processed/brown/rare-ngrams.csv
lm> rare-check tri data/processed/brown/rare-ngrams.csv 10-100
lm> rare-check tri data/processed/brown/rare-ngrams.csv 0-all
lm> quit
```

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

`results/<name>/` contains:

| File | Content |
|---|---|
| `runs.csv` | One row per model x seed: parameters, vocab size, test OOV rate, valid/test perplexity, epochs, train and test time |
| `summary.csv` | Mean and standard deviation per model across seeds |
| `<model>_seed<seed>_history.csv` | Per-epoch train/valid perplexity and time |
| `<model>_seed<seed>.pt` | Best model weights (not committed to git; regenerate with `run.py`) |
| `vocab.json` | Vocabulary list (index = word id) |
| `data_stats.json` | Sentence/token counts per split, vocab size, OOV rates |
| `config.yaml` | Copy of the config used |

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

## Not yet included

- Unsmoothed bigram/trigram and smoothed n-gram baselines.
- Per-sentence probability scoring for the rare/unseen sequence analysis.
- Ablation experiments.
