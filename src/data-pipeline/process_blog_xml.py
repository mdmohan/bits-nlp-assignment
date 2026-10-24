"""Extract blog posts from XML, clean selected artifacts, and write final text.

This links the existing ``extract_blog_posts`` and ``clean_blog_artifacts``
implementations without creating an intermediate folder.

Example (run from the assignment folder)::

    python3 src/process_blog_xml.py \
        --input-dir data/raw/blogs \
        --output-dir data/processed/blog-posts-final

Each source XML becomes a same-named .txt file. Each retained post occupies
one line, and nested input folders are preserved in the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clean_blog_artifacts import clean_text
from extract_blog_posts import extract_posts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract posts from blog XML, clean known artifacts, and write "
            "one final .txt file per source XML."
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
        path for path in input_dir.rglob("*.xml") if path.is_file()
    )
    if not source_files:
        parser.error(f"no .xml files found under {input_dir}")

    totals = {
        "posts_extracted": 0,
        "urlLink_removed": 0,
        "nbsp_removed": 0,
        "amp_converted_to_and": 0,
        "replacement_chars_removed": 0,
        "NULs_removed": 0,
    }
    empty_files = 0
    failures: list[tuple[Path, str]] = []

    for source in source_files:
        try:
            posts = extract_posts(source)
            # Joining first lets the shared cleaner normalize all post text;
            # its line handling preserves one line per extracted post.
            extracted_text = "\n".join(posts)
            cleaned_text, counts = clean_text(extracted_text)

            destination = output_dir / source.relative_to(input_dir).with_suffix(".txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                cleaned_text + ("\n" if cleaned_text else ""), encoding="utf-8"
            )

            totals["posts_extracted"] += len(posts)
            totals["urlLink_removed"] += counts["urlLink"]
            totals["nbsp_removed"] += counts["nbsp"]
            totals["amp_converted_to_and"] += counts["amp_to_and"]
            totals["replacement_chars_removed"] += counts["replacement_char"]
            totals["NULs_removed"] += counts["NUL"]
            empty_files += not cleaned_text
        except (OSError, UnicodeError, ValueError) as exc:
            failures.append((source, str(exc)))

    print(f"Processed {len(source_files) - len(failures)} of {len(source_files)} XML files.")
    print(f"Final text files: {output_dir}")
    print(f"Empty outputs: {empty_files}")
    print("Post and cleanup counts:")
    for key, count in totals.items():
        print(f"  {key}: {count}")
    if failures:
        print(f"Files that could not be processed: {len(failures)}")
        for source, message in failures[:20]:
            print(f"  {source}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
