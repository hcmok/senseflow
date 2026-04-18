import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


def is_valid_word(word, tokenizer):
    # WordNet check
    if not wordnet.synsets(word):
        return False

    # BERT tokenizer check
    tokens = tokenizer.tokenize(word)
    if len(tokens) != 1 or tokens[0] == tokenizer.unk_token:
        return False

    return True


def process_batch(batch, tokenizer):
    """
    Returns list of (word, sentence) tuples
    """
    results = []
    sentence_count = 0
    for item in batch:
        text = item["text"]
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # Normalize whitespace and convert to lowercase
            clean = re.sub(r"\s+", " ", sentence).strip()

            words = clean.split()
            # Count one occurrence per sentence
            unique_words = set(words)
            for word in unique_words:
                # Check if word is valid
                if is_valid_word(word, tokenizer):
                    results.append((word.lower(), clean))
        sentence_count += len(sentences)
    return results, sentence_count


def collect_and_save_samples(
    corpus,
    tokenizer,
    process_sentence_count,
    batch_size,
    min_word_freq,
    max_sample_count_per_word,
    samples_path,
):
    word_freq = Counter()
    word_samples = defaultdict(list)

    dl = DataLoader(
        corpus,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=lambda batch: process_batch(batch, tokenizer),
        pin_memory=True,
    )

    pbar = tqdm(total=process_sentence_count, desc="Processing corpus")
    processed_count = 0
    for batch_results, sentence_count in dl:
        for word, sentence in batch_results:
            word_freq[word] += 1
            if len(word_samples[word]) < max_sample_count_per_word:
                word_samples[word].append(sentence)

        pbar.update(sentence_count)
        processed_count += sentence_count
        if processed_count >= process_sentence_count:
            break

    # Filter words by frequency
    filtered_words = [(w, c) for w, c in word_freq.items() if c >= min_word_freq]
    sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)

    with open(samples_path, "w", encoding="utf-8") as f:
        for word, freq in sorted_words:
            entry = {
                "word": word,
                "freq": freq,
                "samples": word_samples[word],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved samples of {len(sorted_words)} to {samples_path}")


def run_sample_step(cfg):
    samples_path = Path(cfg["data"]["dir"]) / cfg["data"]["samples_path"]

    preprocess_cfg = cfg["preprocess"]

    corpus = load_dataset(
        preprocess_cfg["corpus"]["path"],
        preprocess_cfg["corpus"]["name"],
        split=preprocess_cfg["corpus"]["split"],
        streaming=preprocess_cfg["corpus"]["streaming"],
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["tokenizer_name"])

    process_sentence_count = preprocess_cfg["corpus"]["process_sentence_count"]
    batch_size = preprocess_cfg["batch_size"]
    min_word_freq = preprocess_cfg["vocab"]["min_word_freq"]
    max_sample_count_per_word = preprocess_cfg["vocab"]["max_sample_count_per_word"]

    collect_and_save_samples(
        corpus,
        tokenizer,
        process_sentence_count,
        batch_size,
        min_word_freq,
        max_sample_count_per_word,
        samples_path,
    )


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

    run_sample_step(cfg)


if __name__ == "__main__":
    main(parse_args())
