"""Clean selected extraction artifacts from a folder of blog text files.

Example:
    python3 src/clean_blog_artifacts.py \
        --input-dir data/processed/blog-posts \
        --output-dir data/processed/blog-posts-clean

Each input .txt file is written to the output directory with its relative
path and filename preserved. The input files are never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


URL_LINK_RE = re.compile(r"\burlLink\b", re.IGNORECASE)
NBSP_RE = re.compile(r"&nbsp;|\bnbsp\b", re.IGNORECASE)
# Treat both a decoded standalone `amp` marker and a literal HTML &amp; entity
# as the requested word "and". The entity form must be matched first.
AMP_RE = re.compile(r"&amp;|\bamp\b", re.IGNORECASE)
HORIZONTAL_SPACE_RE = re.compile(r"[ \t\f\v]+")


def clean_text(text: str) -> tuple[str, dict[str, int]]:
    """Remove the requested artifacts and return cleaned text plus counts."""
    counts = {
        "urlLink": len(URL_LINK_RE.findall(text)),
        "nbsp": len(NBSP_RE.findall(text)),
        "amp_to_and": len(AMP_RE.findall(text)),
        "replacement_char": text.count("\ufffd"),
        "NUL": text.count("\x00"),
    }

    # Use spaces for removed markers so neighboring words do not accidentally
    # join (for example, "one\x00two" should not become "onetwo").
    text = URL_LINK_RE.sub(" ", text)
    text = NBSP_RE.sub(" ", text)
    text = AMP_RE.sub(
        lambda match: " and " if match.group().startswith("&") else "and", text
    )
    text = text.replace("\ufffd", " ").replace("\x00", " ")

    # Keep the one-line-per-post format while removing spacing left by markers.
    lines = [HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove urlLink/nbsp/replacement/NUL artifacts and convert amp "
            "markers to 'and' in .txt files."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        parser.error("output directory must be outside the input directory")

    source_files = sorted(
        path for path in input_dir.rglob("*.txt") if path.is_file()
    )
    if not source_files:
        parser.error(f"no .txt files found under {input_dir}")

    totals = {
        "urlLink": 0,
        "nbsp": 0,
        "amp_to_and": 0,
        "replacement_char": 0,
        "NUL": 0,
    }
    for source in source_files:
        # Decode damaged UTF-8 with U+FFFD, then remove those characters below.
        original = source.read_text(encoding="utf-8-sig", errors="replace")
        cleaned, counts = clean_text(original)
        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned + ("\n" if cleaned else ""), encoding="utf-8")
        for key, count in counts.items():
            totals[key] += count

    print(f"Cleaned {len(source_files)} files into {output_dir}")
    print("Replacements/removals:")
    for key, count in totals.items():
        print(f"  {key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
