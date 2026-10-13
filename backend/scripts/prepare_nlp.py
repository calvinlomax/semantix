#!/usr/bin/env python3
"""Install WordNet once; gameplay never downloads NLP resources."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nltk
from dotenv import load_dotenv


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(os.getenv(
        "SEMANTIX_NLTK_DATA", str(root / "data" / "nltk_data")
    )))
    args = parser.parse_args()
    args.output = args.output.expanduser()
    if not args.output.is_absolute():
        args.output = root / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    if not nltk.download("wordnet", download_dir=str(args.output), raise_on_error=True):
        raise SystemExit("Unable to prepare WordNet.")
    nltk.data.path.insert(0, str(args.output))
    from nltk.corpus import wordnet
    wordnet.ensure_loaded()
    print(f"WordNet {wordnet.get_version()} ready at {args.output}")


if __name__ == "__main__":
    main()
