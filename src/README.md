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
│   ├── data.py            # dataset loaders, Corpus, Vocab, batching
│   ├── models.py          # FeedForwardLM, LSTMLM, model registry
│   ├── trainer.py         # generic training loop, evaluation, early stopping, seeding
│   └── controller.py      # runs models x seeds on one dataset and saves results
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

The Brown corpus is downloaded automatically by NLTK on first use (to `~/nltk_data`).
To fetch it manually: `python -c "import nltk; nltk.download('brown')"`.

## Running

From the `source_code` directory:

```sh
~/.venvs/nlp-lm/bin/python -u run.py configs/brown.yaml
```

This trains every model listed in the config for every seed, prints per-epoch progress, and
writes results to `results/<name>/`.

### Using it from Python / a notebook

```python
from lm.data import load_corpus
from lm.controller import LanguageModelController
from lm.trainer import TrainConfig

corpus = load_corpus("brown", categories=["news", "editorial", "reviews"])
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
| `data.loader` | `brown`, `text_files` or `huggingface` (see below) |
| `data.lowercase` | Lowercase all tokens (default `true`) |
| `data.params` | Arguments passed to the loader |
| `vocab.min_freq` | Training words seen fewer times become `<unk>` |
| `vocab.max_size` | Optional cap on vocabulary size |
| `models` | List of `{name, params}`; `name` is a key of `MODELS` in `lm/models.py` |
| `training` | `batch_size`, `epochs`, `lr`, `grad_clip`, `weight_decay`, `patience` |

### Available datasets

| Loader | Params | Notes |
|---|---|---|
| `brown` | `categories` (list or omit for all 15 genres), `ratios`, `split_seed` | Split by whole Brown files (documents) |
| `text_files` | `train`, `valid`, `test` paths | One sentence per line, whitespace-tokenised |
| `huggingface` | `name`, `config`, `text_field`, `splits` | E.g. `name: Salesforce/wikitext`, `config: wikitext-2-raw-v1` |

Example for WikiText-2:

```yaml
data:
  loader: huggingface
  params: {name: Salesforce/wikitext, config: wikitext-2-raw-v1}
```

If the Hugging Face download fails with `File reconstruction error: Byte range not sequential`,
run with `HF_HUB_DISABLE_XET=1`.

### Available models

| Name | Class | Description | Params |
|---|---|---|---|
| `ffnn` | `FeedForwardLM` | Bengio et al. (2003) style: embeddings of the previous `context_size` words are concatenated and passed through one tanh hidden layer | `context_size`, `embed_dim`, `hidden_dim`, `dropout` |
| `lstm` | `LSTMLM` | Word-level LSTM using the whole sentence prefix | `embed_dim`, `hidden_dim`, `num_layers`, `dropout` |

## Extending

**New dataset:** write a function in `lm/data.py` that returns a `Corpus(train, valid, test)`
(each a list of sentences, each sentence a list of tokens) and add it to `LOADERS`.
Use `split_documents(...)` if the data is not already split.

**New model:** write an `nn.Module` in `lm/models.py` with this contract and add it to `MODELS`:

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
