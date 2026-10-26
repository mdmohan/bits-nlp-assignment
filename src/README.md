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

The FFNN's `context_size` is the number of preceding words it receives to predict the next
word. Override it for a controlled context comparison while keeping the same corpus, seed,
and training percentage:

```sh
python -u run.py configs/brown.yaml --algorithm ffnn --context-size 1 --seed 13
python -u run.py configs/brown.yaml --algorithm ffnn --context-size 2 --seed 13
python -u run.py configs/brown.yaml --algorithm ffnn --context-size 4 --seed 13
```

Context size `1` gives a bigram-style neural context; `2` gives a trigram-style context.
Each override writes to its own `results/<name>_ffnn_ctx<N>/` folder, and the context size is
recorded in `runs.csv`, `summary.csv`, and the saved config. The option applies to FFNN only.

To run several FFNN context sizes in one experiment, add a top-level list beside `seeds` in
the YAML config:

```yaml
seeds: [13, 42, 2026]
context_sizes: [1, 2, 4]
```

Each configured FFNN is trained once per context size and seed; other configured models run
once per seed. Checkpoints and histories include the context size in their filenames, and the
run table reports it as a separate result column. `--context-size N` still overrides the list
for a one-context run.

For a quicker run, pass `--train-percent` to use only the first portion of the training split;
validation and test data remain complete:

```sh
python -u run.py configs/brown.yaml --train-percent 10
```

The default is `100`, which uses all training sentences. The chosen percentage is recorded in
the output config, data statistics, and run table.

### Inspecting a saved run

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

The evaluation table shows each context, expected next word, n-gram prediction, and the
probability assigned to the expected word. The default table shows the first 100 events;
`--limit 0` prints every event. Perplexity and top-1 accuracy summarize the full file.

To compare against a neural checkpoint, load it and compare both models on the same test file:

```text
lm> nn load src/results/brown_press_neural/ffnn_seed13.pt
lm> vocab 10
lm> predict the cat
lm> ngram brown_bigram --n 2 --train data/processed/brown/train.txt
lm> compare --one ngram brown_bigram --two neural ffnn_seed13.pt --test data/processed/brown/test.txt
lm> quit
```

When the n-gram is created from the neural model's exact training file, the CLI uses the
checkpoint's saved vocabulary, lowercasing rule, and training percentage. Otherwise it builds
a vocabulary from the supplied training text (`--min-freq 2` by default). N-gram models are
kept in memory for the current CLI session; recreate them after restarting the CLI. A zero
probability event gives an unsmoothed model infinite perplexity.

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
