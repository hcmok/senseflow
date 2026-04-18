import argparse
import json
from pathlib import Path
from typing import List, Set, Tuple

import yaml
from nltk.corpus import wordnet

RELATION_CODES = {
    "synonym": 0,
    "antonym": 1,
    "hypernym": 2,
    "hyponym": 3,
    "meronym": 4,
    "holonym": 5,
    "derivation": 6,
    "entailment": 7,
    "cause": 8,
}


def add_relation_pair(pairs, w1, w2, code, vocab):
    if w1 != w2 and w1 in vocab and w2 in vocab:
        pairs.add((w1, w2, code))


def collect_and_save_pairs(vocab, pairs_path):
    pairs: Set[Tuple[str, str, int]] = set()
    for word in vocab:
        # Get all synsets for the word (words not in WordNet will be skipped)
        synsets: List = wordnet.synsets(word)
        for synset in synsets:
            for lemma in synset.lemmas():
                # Synonyms
                synonym = lemma.name().lower()
                add_relation_pair(
                    pairs, word, synonym, RELATION_CODES["synonym"], vocab
                )

                # Antonyms
                for ant in lemma.antonyms():
                    antonym = ant.name().lower()
                    add_relation_pair(
                        pairs, word, antonym, RELATION_CODES["antonym"], vocab
                    )

                # Derivational forms
                for der in lemma.derivationally_related_forms():
                    related_der = der.name().lower()
                    add_relation_pair(
                        pairs, word, related_der, RELATION_CODES["derivation"], vocab
                    )

            # Hypernyms
            for hyper in synset.hypernyms():
                for lemma in hyper.lemmas():
                    add_relation_pair(
                        pairs,
                        word,
                        lemma.name().lower(),
                        RELATION_CODES["hypernym"],
                        vocab,
                    )

            # Hyponyms
            for hypo in synset.hyponyms():
                for lemma in hypo.lemmas():
                    add_relation_pair(
                        pairs,
                        word,
                        lemma.name().lower(),
                        RELATION_CODES["hyponym"],
                        vocab,
                    )

            # Meronyms
            for mer in (
                synset.part_meronyms()
                + synset.substance_meronyms()
                + synset.member_meronyms()
            ):
                for lemma in mer.lemmas():
                    add_relation_pair(
                        pairs,
                        word,
                        lemma.name().lower(),
                        RELATION_CODES["meronym"],
                        vocab,
                    )

            # Holonyms
            for hol in (
                synset.part_holonyms()
                + synset.substance_holonyms()
                + synset.member_holonyms()
            ):
                for lemma in hol.lemmas():
                    add_relation_pair(
                        pairs,
                        word,
                        lemma.name().lower(),
                        RELATION_CODES["holonym"],
                        vocab,
                    )

            # Entailments (verbs)
            for ent in synset.entailments():
                for lemma in ent.lemmas():
                    add_relation_pair(
                        pairs,
                        word,
                        lemma.name().lower(),
                        RELATION_CODES["entailment"],
                        vocab,
                    )

            # Causes (verbs)
            for cause in synset.causes():
                for lemma in cause.lemmas():
                    add_relation_pair(
                        pairs,
                        word,
                        lemma.name().lower(),
                        RELATION_CODES["cause"],
                        vocab,
                    )

    with open(pairs_path, "w", encoding="utf-8") as f:
        f.write("word1,word2,relation\n")
        for w1, w2, code in pairs:
            f.write(f"{w1},{w2},{code}\n")
    print(f"Saved pairs to {pairs_path}")


def run_pair_step(cfg):
    pairs_path = Path(cfg["data"]["dir"]) / cfg["data"]["pairs_path"]
    embedding_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["embedding_metadata_path"]
    )

    with open(embedding_metadata_path, "r") as f:
        embedding_metadata = json.load(f)
    words = embedding_metadata["word_to_embedding_indices"].keys()
    words = set(words)

    collect_and_save_pairs(words, pairs_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_pair_step(cfg)


if __name__ == "__main__":
    main(parse_args())
