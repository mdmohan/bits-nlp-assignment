# Colab Data-Efficiency Pilot

[nltk_data_efficiency.ipynb](nltk_data_efficiency.ipynb) runs the existing model
implementations against audited Gutenberg, WebText and Inaugural inputs on Colab.
No access token or password belongs in this notebook or in Git.

## Prerequisites

- Install the local environment using the [root setup](../README.md#start-here).
- Restore the imported corpus files described in the [prepared-data guide](../data/processed/readme.txt),
  including the matching import manifest, document stores and checksum sidecars.
- Keep the prepared input bundle private. It contains corpus text as SQLite
  sentence indexes and is intentionally ignored by Git.
- Use a GPU runtime with sufficient memory and Google Drive space for several
  gigabytes of checkpoints and result archives. Colab limits still apply.

The synthetic tests and `--dry-run` do not need private corpora. Building a real
bundle does. A Git clone alone is not the complete experiment input.

## Build A Matching Bundle

From the repository root:

```sh
~/.venvs/nlp-lm/bin/python src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage all --dry-run
~/.venvs/nlp-lm/bin/python -u src/run.py --suite src/configs/data_efficiency_nltk_pilot.yaml --stage prepare
PYTHONPATH=src ~/.venvs/nlp-lm/bin/python -u -m lm.colab --suite src/configs/data_efficiency_nltk_pilot.yaml --output colab/nltk_pilot_bundle.zip
```

Preparation preserves imported test membership and derives validation only from
existing training documents. The explicit WebText repair keeps test bytes
unchanged and excludes overlapping training source groups in a separate derived
directory. A blocked corpus is not silently replaced with a sentence-sampled run.

The packager writes `nltk_pilot_bundle.zip` and a `.zip.sha256` sidecar. It includes
source/tests, the selected corpora's prepared manifests and sentence indexes,
configuration, repair audit and `bundle_manifest.json`. Individual bundle members
have SHA-256 hashes. It does not include original corpus databases or checkpoints.

The checked-in notebook's `EXPECTED_BUNDLE_SHA256` identifies the archived pilot
bundle, not every future build. **After rebuilding, set it to the new sidecar's
hash.** ZIP metadata or source changes can change the archive hash even when
settings look similar. Do not bypass the check or assume an old bundle contains
newly edited code.

If source or settings changed since a previous preparation, select a new `name`
and `output_root` in a copied suite configuration before running it. Supply that
same configuration to both commands. Completed output identities are immutable.

## Run The Notebook

1. Upload the notebook to Google Colab and select **Runtime > Change runtime type
   > T4 GPU**.
2. Set `EXPECTED_BUNDLE_SHA256` to the bundle's sidecar hash. Keep `EPOCHS = 3`,
   `SEEDS = [42]` and the three selected corpora for this pilot. Use a new
   `OUTPUT_NAME` for a new bundle, changed settings or a new experiment.
3. Run the upload cell and select exactly one prepared ZIP. The notebook verifies
   the outer hash, safe extraction paths, member hashes and corpus selection.
4. Run the Drive cell and complete Google's authorization directly in the
   browser. Keep `USE_DRIVE = True` for durable outputs. Without Drive, results
   under `/content` disappear when the runtime is discarded.
5. Run the stage cell: `train`, `seal`, `evaluate`, then `report`. It saves combined
   output to `colab_execution.log` and stops on a failed stage.
6. Run the preview and export cells. The export writes an artifact hash manifest,
   creates a ZIP outside the result folder, retains it beside the result folder
   on Drive, and requests a browser download.

The default Drive location is
`MyDrive/NLP_Experiments/<OUTPUT_NAME>`. The archived completed run is
`nltk_data_efficiency_pilot_v2`; do not reuse that directory for changed code.

## Scientific Settings

| Setting | Pilot |
| --- | --- |
| Training budgets | 10%, 20%, ..., 100% of whole training documents, nested and seeded |
| Validation/test | Fixed raw populations per corpus; test scoring follows sealing |
| Vocabulary | Training-only, fresh per budget, `min_freq=2`, shared within that budget |
| Models | `bigram_mle`, `trigram_mle`, `trigram_addk`, `lstm`, `ffnn` |
| Add-k selection | Validation grid: 0.01, 0.1, 1.0 |
| Neural training | Three epochs, final seed 42, fresh model at every budget |
| Prediction targets | Word tokens plus EOS; PAD/BOS excluded |
| Matrix | 150 final fits, including 60 neural fits, plus 90 add-k tuning fits |

The FFNN uses two preceding words; the LSTM uses the sentence prefix. A saved
best-validation `.pt` contains weights only, not optimizer state. WebText's
rounded budget aliases and retained within-source variants must remain labelled.
Cross-budget perplexity is not a fixed-vocabulary intervention.

## Resume And Failures

Rerun the same notebook cells with the same bundle, settings and `OUTPUT_NAME`
after a recoverable interruption. A replacement runtime needs the upload and
Drive cells again. Complete fits are skipped only after their fingerprints and
artifact hashes pass; incomplete fits restart in new preserved attempt folders.
This is not exact mid-epoch optimizer resume. Environment changes can invalidate
a fit identity; use a new output directory rather than forcing reuse.

Check `colab_execution.log`, per-run `status.json` and `attempts/*/run.log` for the
failing stage. Do not change seeds, resource limits or model sizes inside an
existing run to hide an error. For quota/authentication failures, restore access
or start a separate eligible runtime; repeated automatic retries cannot remove
Colab limits. Increasing cached GPU usage alone does not establish a memory leak.

If Colab's output JavaScript fails, inspect the durable log through Drive. A blank
preview does not prove training failed. The completed pilot used a direct Drive
folder download because preview/export controls were unreliable.

## Retrieve And Verify All Results

Download the exported ZIP, or download the entire result folder through Drive.
Drive can split a folder into `-001.zip`, `-002.zip`, `-003.zip`, etc. Keep every
part, wait for the downloads to finish, and inspect each outer ZIP index before
extracting. File existence or `zipfile.is_zipfile()` alone is insufficient during
an active transfer: nested PyTorch checkpoint ZIPs can cause false positives.

Extract under a new `results/colab_imports/<timestamp>/` directory, never over a
previous run. Reject unsafe paths, symlinks and duplicate file names. Check ZIP
CRCs, the artifact manifest if present, sealed artifact hashes from `seal.json`
and each run's `status.json`, plus test hashes in `evaluation_status.json`. Preserve
the original ZIPs; verification does not require unpickling model/count files.

Expect 150 final `result.json`, validation metrics and test metrics, 90 tuning
results, 60 neural histories/checkpoints and all budget vocabularies. Keep the
input bundle: its prepared sentence indexes and manifests preserve the exact
train/validation/test data. Histories and fitting statistics are included;
separate final training-set rescoring was not performed.

Open the extracted `report.html` with its relative assets intact. Table and CSV
model order is `bigram_mle`, `trigram_mle`, `trigram_addk`, `lstm`, `ffnn` within
each budget. See the [source guide](../src/README.md#saved-artifacts) for artifacts
and the [batch report](../docs/research_directions/data_efficiency/batch-report.yaml)
for the verified pilot. Generated results and private bundles are not pushed.