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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpora", nargs="+")
    args = parser.parse_args()
    create_bundle(ExperimentSuite(args.suite, args.corpora), args.output)


if __name__ == "__main__":
    main()