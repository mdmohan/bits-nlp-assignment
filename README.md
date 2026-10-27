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

Keep the virtual environment outside cloud-synced folders. With plain `pip`,
create the environment using `python3.12 -m venv ~/.venvs/nlp-lm`, then run
`~/.venvs/nlp-lm/bin/python -m pip install -r src/requirements.txt`.

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