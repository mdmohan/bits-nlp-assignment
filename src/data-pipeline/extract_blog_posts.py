"""Extract each ``<post>`` from blog XML files into one line per post.

Example:
    python3 src/extract_blog_posts.py \
        --input-dir data/raw/blogs \
        --output-dir data/processed/blogs

The output keeps the source filename, changing only ``.xml`` to ``.txt``.
HTMLParser is used because some corpus files are not well-formed XML.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re


class BlogPostParser(HTMLParser):
    """Collect text inside each post element, tolerating imperfect markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.posts: list[str] = []
        self._in_post = False
        self._depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "post":
            if self._in_post:
                self._depth += 1
            else:
                self._in_post = True
                self._depth = 1
                self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "post" or not self._in_post:
            return
        self._depth -= 1
        if self._depth <= 0:
            post = re.sub(r"\s+", " ", " ".join(self._parts)).strip()
            if post:
                self.posts.append(post)
            self._in_post = False
            self._depth = 0
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._in_post:
            self._parts.append(data)

    def close(self) -> None:
        super().close()
        # Keep useful text if a malformed source file has an unclosed <post>.
        if self._in_post:
            post = re.sub(r"\s+", " ", " ".join(self._parts)).strip()
            if post:
                self.posts.append(post)
            self._in_post = False


def extract_posts(source: Path) -> list[str]:
    # Replace undecodable bytes instead of failing the entire blog file.
    text = source.read_bytes().decode("utf-8", errors="replace")
    parser = BlogPostParser()
    parser.feed(text)
    parser.close()
    return parser.posts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write each XML blog post as one line in a same-named .txt file."
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

    sources = sorted(input_dir.glob("*.xml"))
    if not sources:
        parser.error(f"no .xml files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    empty = 0
    for source in sources:
        posts = extract_posts(source)
        destination = output_dir / f"{source.stem}.txt"
        destination.write_text("\n".join(posts) + ("\n" if posts else ""), encoding="utf-8")
        written += 1
        empty += not posts

    print(f"Wrote {written} text files to {output_dir} ({empty} with no extracted posts).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
