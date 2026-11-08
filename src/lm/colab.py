"""Package selected, audited prepared corpora for a private Colab session."""

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path

import yaml

from .experiments import ExperimentSuite
from .preparation import file_sha256


def create_bundle(suite, destination):
    statuses = suite.preflight(prepare=True)
    blocked = {name: state for name, state in statuses.items() if state["status"] != "prepared"}
    if blocked:
        raise ValueError(f"Selected corpora are not eligible for packaging: {blocked}")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    config = copy.deepcopy(suite.config)
    config["corpora"] = {name: config["corpora"][name] for name in suite.corpora}
    config["prepared_data"] = {"rebuild": False, "packaged_only": True, "index_root": "input/prepared"}
    code_paths = sorted((suite.root / "src" / "lm").rglob("*.py"))
    code_paths.extend(sorted((suite.root / "src" / "tests").glob("*.py")))
    code_paths.extend([suite.root / "src" / "run.py", suite.root / "src" / "requirements.txt"])
    hashes = {}

    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def add_bytes(name, data):
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()

        def add_file(path, name):
            archive.write(path, name)
            hashes[name] = file_sha256(path)

        for path in code_paths:
            add_file(path, path.relative_to(suite.root).as_posix())
        for corpus in suite.corpora:
            manifest = suite._prepared(corpus)
            index = Path(manifest["index"])
            manifest["index"] = "sentences.sqlite"
            add_bytes(f"input/prepared/{corpus}/manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
            add_file(index, f"input/prepared/{corpus}/sentences.sqlite")
            audit = suite.output / "curated" / corpus / "repair.json"
            if audit.exists():
                add_file(audit, f"input/audits/{corpus}_repair.json")
        add_file(suite.config_path, "input/original_suite.yaml")
        add_bytes("input/suite.yaml", yaml.safe_dump(config, sort_keys=False).encode())
        add_bytes("bundle_manifest.json", (json.dumps({"schema_version": 1, "files": hashes,
                                                       "corpora": suite.corpora, "source_code_hashes": suite.code_hashes}, indent=2, sort_keys=True) + "\n").encode())
    temporary.replace(destination)
    digest = file_sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(digest + "  " + destination.name + "\n", encoding="ascii")
    print(f"Bundle: {destination} ({destination.stat().st_size / 1024 ** 2:.2f} MiB)")
    print(f"SHA256: {digest}")
    return destination


def create_notebook(suite, bundle, destination):
    """Generate a notebook pinned to the bundled protocol, without changing it."""
    bundle = Path(bundle).resolve()
    with zipfile.ZipFile(bundle) as archive:
        config = yaml.safe_load(archive.read("input/suite.yaml"))
    neural_seeds = [spec["final_seeds"] for spec in config["models"] if spec["family"] == "neural"]
    if not neural_seeds or any(seeds != neural_seeds[0] for seeds in neural_seeds):
        raise ValueError("Notebook generation requires the same final seeds for all neural models.")
    template = suite.root / "colab" / "nltk_data_efficiency.ipynb"
    notebook = json.loads(template.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        f"# {config['name']}\n", "\n",
        "Fresh LSTM and FFNN fits; validation selects the restored best checkpoint. "
        "A patience stop is a validation plateau; a run reaching the epoch ceiling is budget-limited, not proven converged.\n",
        "\n",
        "Run cells in order on a T4 GPU. Authorize Google Drive directly in the browser. "
        "The original pilot and imports remain unchanged. Prepared manifests contain the derived quarantine audit.\n",
        "\n",
        "The bundle checksum and training settings below are frozen together. "
        "Do not alter them to resume a different experiment. Cross-budget vocabularies change prediction support.\n",
    ]
    settings = (
        "from pathlib import Path\n\n"
        f"EPOCHS = {config['training']['epochs']!r}\n"
        f"PATIENCE = {config['training']['patience']!r}\n"
        f"BUDGETS = {config['budgets_percent']!r}\n"
        f"SEEDS = {neural_seeds[0]!r}\n"
        f"CORPORA = {list(config['corpora'])!r}\n"
        "USE_DRIVE = True\nREQUIRE_GPU = True\n"
        f"OUTPUT_NAME = {config['name']!r}\n"
        'WORKSPACE = Path("/content/nlp_assignment")\n'
        f"EXPECTED_BUNDLE_SHA256 = {file_sha256(bundle)!r}\n\n"
        "assert EPOCHS > 0 and PATIENCE > 0\n"
        "assert CORPORA and SEEDS and BUDGETS\n"
        "assert Path(OUTPUT_NAME).name == OUTPUT_NAME\n"
        'print("Experiment:", CORPORA, "epochs:", EPOCHS, "seeds:", SEEDS)\n'
    )
    notebook["cells"][1]["source"] = settings.splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            compile("".join(cell["source"]), "<colab-cell>", "exec")
    destination = Path(destination).resolve()
    if destination == template.resolve():
        raise ValueError("Do not overwrite the archived pilot notebook; choose a new notebook path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Matching notebook: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpora", nargs="+")
    parser.add_argument("--notebook", help="Write a matching, checksum-pinned notebook to this new path.")
    args = parser.parse_args()
    suite = ExperimentSuite(args.suite, args.corpora)
    bundle = create_bundle(suite, args.output)
    if args.notebook:
        create_notebook(suite, bundle, args.notebook)


if __name__ == "__main__":
    main()