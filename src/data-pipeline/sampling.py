"""Deterministic document-level sampling before corpus cleaning and loading."""

from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
from pathlib import Path


def normalize_sampling(value) -> dict | None:
    """Validate manifest sampling settings; accept mapping or one-key mapping list."""
    if value is None or value == {} or value == []:
        return None
    if isinstance(value, list):
        merged = {}
        for item in value:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError("sampling list entries must each contain one key/value pair")
            key, item_value = next(iter(item.items()))
            if key in merged:
                raise ValueError(f"sampling contains duplicate setting {key!r}")
            merged[key] = item_value
        value = merged
    if not isinstance(value, dict):
        raise ValueError("sampling must be an object with sample-percent and sample-seed")
    unknown = set(value) - {"sample-percent", "sample-seed"}
    if unknown:
        raise ValueError(f"unknown sampling setting(s): {', '.join(sorted(unknown))}")
    if not value:
        return None
    if "sample-percent" not in value or "sample-seed" not in value:
        raise ValueError("sampling requires both sample-percent and sample-seed")

    percent = value["sample-percent"]
    seed = value["sample-seed"]
    if isinstance(percent, bool) or not isinstance(percent, (int, float)):
        raise ValueError("sampling.sample-percent must be a number greater than 0 and at most 100")
    if not math.isfinite(float(percent)) or not 0 < float(percent) <= 100:
        raise ValueError("sampling.sample-percent must be greater than 0 and at most 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("sampling.sample-seed must be an integer")
    return {"sample-percent": float(percent), "sample-seed": seed}


def sample_documents(
    source_root: Path,
    documents: list[Path],
    destination: Path,
    sampling: dict,
) -> tuple[Path, dict]:
    """Link a seeded percentage of document files into a temporary source tree."""
    source_root = source_root.resolve()
    documents = sorted((path.resolve() for path in documents), key=lambda path: path.as_posix())
    if not documents:
        raise ValueError(f"No source documents are eligible for sampling under {source_root}")
    if any(not path.is_relative_to(source_root) for path in documents):
        raise ValueError("A sampling input file is outside the source folder")

    percent = sampling["sample-percent"]
    seed = sampling["sample-seed"]
    selected_count = min(len(documents), math.ceil(len(documents) * percent / 100))
    if selected_count == len(documents):
        selected = documents
    else:
        selected = sorted(
            random.Random(seed).sample(documents, selected_count),
            key=lambda path: path.as_posix(),
        )

    destination = destination.resolve()
    if destination == source_root or source_root in destination.parents:
        raise ValueError("Sampling output must be outside the extracted source folder")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    selected_ids = []
    for source in selected:
        relative = source.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(source, target)
        except OSError:
            shutil.copy2(source, target)
        selected_ids.append(relative.as_posix())

    digest = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    metadata = {
        "config": sampling,
        "source_document_count": len(documents),
        "sampled_document_count": selected_count,
        "selection_sha256": digest,
    }
    print(
        f"Document sample: selected {selected_count:,}/{len(documents):,} files "
        f"({percent:g}%; seed={seed})",
        flush=True,
    )
    return destination, metadata
