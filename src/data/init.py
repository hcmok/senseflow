import argparse
from pathlib import Path

import nltk
import yaml


def run_init_step(cfg):
    # Download NLTK resources
    try:
        from nltk.corpus import wordnet

        wordnet.synsets("test")
        print("WordNet is available.")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")
    try:
        from nltk.tokenize import sent_tokenize

        sent_tokenize("test")
        print("Punkt tokenizer is available.")
    except LookupError:
        nltk.download("punkt_tab")

    # Create data directory
    data_dir = Path(cfg["data"]["dir"])
    data_dir.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_init_step(cfg)


if __name__ == "__main__":
    main(parse_args())
