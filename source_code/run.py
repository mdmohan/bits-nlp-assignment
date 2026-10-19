"""Command-line entry point.

    python run.py configs/brown.yaml
"""

import argparse

from lm.controller import run_from_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare neural language models.")
    parser.add_argument("config", help="path to a YAML experiment config")
    run_from_config(parser.parse_args().config)
