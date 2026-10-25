import argparse
import shelve
from pathlib import Path
from corpus_parser import create_parser
from log_utils import display_path


def collect_files(corpus_path: Path, extension: str) -> list[Path]:
    """Resolves single file orRecursively gathers matching files from directory."""
    if corpus_path.is_file():
        return [corpus_path]
    if corpus_path.is_dir():
        # Ensure extension starts with a dot
        if extension:
            ext = extension if extension.startswith('.') else f".{extension}"
            files = sorted(path for path in corpus_path.rglob(f"*{ext}") if path.is_file())
        else:
            files = sorted(path for path in corpus_path.rglob("*") if path.is_file())
        
        if not files:
            raise FileNotFoundError(f"No files matching '{ext}' found in {corpus_path}")
        return files
    raise FileNotFoundError(f"Path does not exist: {corpus_path}")


def main():
    parser_cli = argparse.ArgumentParser(
        description="Parse corpus files, store structured output in DB, and dump sentences."
    )
    parser_cli.add_argument(
        "corpus_path",
        type=Path,
        help="Path to a text file or directory containing corpus files."
    )
    parser_cli.add_argument(
        "--corpus-type", "-c",
        type=str,
        default="gutenberg",
        help="Corpus parser type (e.g., 'gutenberg', 'brown', 'reuters', 'webtext'). Default: 'gutenberg'."
    )
    parser_cli.add_argument(
        "--db-path", "-d",
        type=Path,
        default=Path("parsed_corpus.db"),
        help="Path for the output database file. Default: 'parsed_corpus.db'."
    )
    parser_cli.add_argument(
        "--dump-path", "-o",
        type=Path,
        help="Output text file path for dumping all assembled sentences. Default: 'all_sentences.txt'."
    )
    parser_cli.add_argument(
        "--ext", "-e",
        type=str,
        default="",
        help="File extension to match when corpus_path is a directory. Default: 'none'."
    )
    parser_cli.add_argument(
        "--verbose",
        action="store_true",
        help="Print one progress line per source document.",
    )

    args = parser_cli.parse_args()
    

    # 1. Instantiate appropriate parser object via factory
    corpus_parser = create_parser(args.corpus_type)
    
    # 2. Collect input files
    target_files = collect_files(args.corpus_path, args.ext)
    print(f"Found {len(target_files)} file(s) to process.")

    total_paragraphs = 0
    total_sentences = 0

    # 3. Process files, write to persistent DB and stream sentences
    # Using shelve for pythonic dict interface: db[file_path_str] = result
    with shelve.open(str(args.db_path)) as db:
        sentence_dump = None
        if args.dump_path:
            sentence_dump = open(args.dump_path, "w", encoding="utf-8")

        for idx, file_path in enumerate(target_files, start=1):
            if 'README' in str(file_path) or 'CONTENT' in str(file_path):
                print("Skip", file_path.name)
                continue
            key = str(file_path.resolve())
            
            try:
                # Parse file
                result = corpus_parser.parse(file_path)
                
                # Store structured dictionary into database: db[file_path] = result
               
                db[key] = result
                file_sentence_count = 0
                for paragraph in result:
                    for sentence in paragraph.get("sentences", []):
                        clean_sentence = sentence.strip()
                        if clean_sentence:
                            if sentence_dump:
                                # Write sentences to single corpus file (1 sentence per line)
                                sentence_dump.write(clean_sentence + "\n")
                            file_sentence_count += 1

                total_paragraphs += len(result)
                total_sentences += file_sentence_count

                if args.verbose:
                    print(f"[{idx}/{len(target_files)}] {file_path.name}: {file_sentence_count:,} sentences")

            except Exception as e:
                print(f"[{idx}/{len(target_files)}] Error processing {file_path.name}: {e}")

    print("\n--- Pipeline Complete ---")
    print(f"Database saved to : {display_path(args.db_path)}")
    if args.dump_path:
        print(f"Sentences dumped to: {display_path(args.dump_path)}")
    print(f"Total Paragraphs  : {total_paragraphs}")
    print(f"Total Sentences   : {total_sentences}")


if __name__ == "__main__":
    main()
