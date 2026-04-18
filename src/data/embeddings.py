import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_possible_samples(path):
    possible_sample_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            possible_sample_count += len(obj["samples"])
    return possible_sample_count


def build_regex(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            words.append(obj["word"])
    return {w: re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words}


def generate_word_embeddings(
    samples_path, tokenizer, model, regexes, batch_size, device
):
    """
    Generator that yields (word, list_of_embeddings)
    """
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            word = obj["word"]
            sentences = obj["samples"]
            word_sample_embeddings = []

            # Use regex to find whole words to avoid substring matches (e.g. 'cat' in 'catch')
            pattern = regexes[word]

            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i : i + batch_size]

                inputs = tokenizer(
                    batch_sentences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                with torch.inference_mode():
                    outputs = model(**inputs)
                    hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

                for j, sentence in enumerate(batch_sentences):
                    match = pattern.search(sentence)
                    if not match:
                        print(f"Word '{word}' not found in sentence: {sentence}")
                        continue

                    start, end_exclusive = match.span()
                    end = end_exclusive - 1

                    s_idx = inputs.char_to_token(
                        batch_or_char_index=j, char_index=start
                    )
                    e_idx = inputs.char_to_token(batch_or_char_index=j, char_index=end)

                    if s_idx is None or e_idx is None:
                        # Happens if the word is past the 512 token truncation limit
                        continue

                    # Extract and average sub-word embeddings
                    sub_word_embedding = hidden[j, s_idx : e_idx + 1, :]
                    word_embedding = sub_word_embedding.mean(dim=0).cpu().numpy()
                    word_sample_embeddings.append(word_embedding)

            if word_sample_embeddings:
                yield word, np.vstack(word_sample_embeddings)


def compute_and_save_embeddings(
    tokenizer,
    model,
    original_dim,
    pca_n_components,
    batch_size,
    samples_path,
    raw_embeddings_path,
    embeddings_path,
    embedding_metadata_path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Compute raw embeddings
    possible_sample_count = count_possible_samples(samples_path)

    raw_embeddings_memmap = np.memmap(
        raw_embeddings_path,
        dtype=np.float32,
        mode="w+",
        shape=(possible_sample_count, original_dim),
    )

    raw_sum = None
    raw_sum_sq = None
    sample_count = 0
    offset = 0
    word_to_embedding_indices = {}

    total_lines = count_lines(samples_path)
    regexes = build_regex(samples_path)

    generator = generate_word_embeddings(
        samples_path, tokenizer, model, regexes, batch_size, device
    )
    for word, emb_arr in tqdm(generator, total=total_lines):
        if raw_sum is None:
            dim = emb_arr.shape[1]
            raw_sum = np.zeros(dim)
            raw_sum_sq = np.zeros(dim)
        raw_embeddings_memmap[offset : offset + len(emb_arr)] = emb_arr
        word_to_embedding_indices[word] = [offset, offset + len(emb_arr)]
        raw_sum += np.sum(emb_arr, axis=0)
        raw_sum_sq += np.sum(emb_arr**2, axis=0)
        offset += len(emb_arr)
        sample_count += len(emb_arr)

    raw_embeddings_memmap.flush()

    if raw_sum is None or raw_sum_sq is None:
        raise ValueError("Cannot compute global mean and std")

    global_mean = raw_sum / sample_count
    global_std = np.sqrt((raw_sum_sq / sample_count) - global_mean**2 + 1e-8)

    # Fit incremental PCA
    print("Fit incremental PCA")

    batch_size = 4096
    assert batch_size > pca_n_components  # Ensure buffer size > pca_n_components

    ipca = IncrementalPCA(n_components=pca_n_components)

    # Skip last chunk of size < pca_n_components
    for i in range(0, sample_count, batch_size):
        end = min(i + batch_size, sample_count)
        # Z-score normalization
        chunk = (raw_embeddings_memmap[i:end] - global_mean) / global_std
        if len(chunk) >= pca_n_components:
            ipca.partial_fit(chunk)

    explained_var = np.cumsum(ipca.explained_variance_ratio_)
    total_explained_variance = explained_var[-1]
    print(
        f"Explained variance by {pca_n_components} components: {total_explained_variance:.4f}"
    )

    # Transform embeddings and write reduced embeddings to disk
    print("Transform embeddings and write reduced embeddings to disk")
    embeddings_memmap = np.memmap(
        embeddings_path,
        dtype=np.float32,
        mode="w+",
        shape=(sample_count, pca_n_components),
    )

    for i in range(0, sample_count, batch_size):
        end = min(i + batch_size, sample_count)
        chunk = (raw_embeddings_memmap[i:end] - global_mean) / global_std
        embeddings_memmap[i:end] = ipca.transform(chunk)

    embeddings_memmap.flush()
    global_mean = np.mean(embeddings_memmap, axis=0)
    global_std = np.std(embeddings_memmap, axis=0)

    output = {
        "dimension": pca_n_components,
        "embedding_count": sample_count,
        "global_embedding_mean": global_mean.tolist(),
        "global_embedding_std": global_std.tolist(),
        "total_explained_variance": total_explained_variance,
        "dtype": "float32",
        "word_to_embedding_indices": word_to_embedding_indices,
    }

    with open(embedding_metadata_path, "w") as f:
        json.dump(output, f)
    print(
        f"Saved embeddings to {embeddings_path} and metadata to {embedding_metadata_path}"
    )


def run_embedding_step(cfg):
    samples_path = Path(cfg["data"]["dir"]) / cfg["data"]["samples_path"]
    raw_embeddings_path = Path(cfg["data"]["dir"]) / cfg["data"]["raw_embeddings_path"]
    embeddings_path = Path(cfg["data"]["dir"]) / cfg["data"]["embeddings_path"]
    embedding_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["embedding_metadata_path"]
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["tokenizer_name"])
    model = AutoModel.from_pretrained(cfg["encoder"]["model_name"])

    original_dim = cfg["preprocess"]["embeddings"]["original_dim"]
    pca_n_components = cfg["preprocess"]["embeddings"]["pca_n_components"]
    batch_size = cfg["preprocess"]["batch_size"]

    compute_and_save_embeddings(
        tokenizer,
        model,
        original_dim,
        pca_n_components,
        batch_size,
        samples_path,
        raw_embeddings_path,
        embeddings_path,
        embedding_metadata_path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_embedding_step(cfg)


if __name__ == "__main__":
    main(parse_args())
