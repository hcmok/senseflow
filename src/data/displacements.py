import argparse
import json
from pathlib import Path

import numpy as np
import ot
import pandas as pd
import torch
import yaml
from scipy.spatial.distance import cdist
from tqdm import tqdm


def compute_and_save_displacements(
    pairs_df,
    embeddings_path,
    embedding_metadata,
    centroids_path,
    centroid_metadata,
    displacements_path,
    displacement_indices_path,
    displacement_metadata_path,
    max_sample_count_per_word,
    reg,
    max_iter,
    stop_threshold,
    device,
):
    """
    Compute displacement vectors from source_cloud to target_cloud using Sinkhorn OT.
    """

    d = embedding_metadata["dimension"]
    embedding_count = embedding_metadata["embedding_count"]
    word_to_centroid_indices = centroid_metadata["word_to_centroid_indices"]
    word_sample_to_centroid_indices = centroid_metadata[
        "word_sample_to_centroid_indices"
    ]
    word_to_embedding_indices = embedding_metadata["word_to_embedding_indices"]
    centroid_count = centroid_metadata["centroid_count"]
    embeddings_memmap = np.memmap(
        embeddings_path, dtype=np.float32, mode="r", shape=(embedding_count, d)
    )
    centroids_memmap = np.memmap(
        centroids_path, dtype=np.float32, mode="r", shape=(centroid_count, d)
    )
    n_rows = len(pairs_df)
    # Since only one centroid is used to represent each word, the number of displacement vectors may be smaller
    displacements_memmap = np.memmap(
        displacements_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_rows * max_sample_count_per_word, d),
    )
    displacement_items = []

    global_embedding_mean = torch.tensor(
        embedding_metadata["global_embedding_mean"], device=device, dtype=torch.float32
    )
    global_embedding_std = torch.tensor(
        embedding_metadata["global_embedding_std"], device=device, dtype=torch.float32
    )
    eps = 1e-8

    index = 0

    for _, row in tqdm(
        pairs_df.iterrows(), total=len(pairs_df), desc="Processing pairs"
    ):
        w1 = row["word1"]
        w2 = row["word2"]
        w1_centroids = centroids_memmap[word_to_centroid_indices[w1]]
        w2_centroids = centroids_memmap[word_to_centroid_indices[w2]]
        sim_matrix = 1 - cdist(w1_centroids, w2_centroids, metric="cosine")
        idx1_local, idx2_local = np.unravel_index(
            np.argmax(sim_matrix), sim_matrix.shape
        )
        idx1_global = word_to_centroid_indices[w1][idx1_local]
        idx2_global = word_to_centroid_indices[w2][idx2_local]
        # Filter samples that belong to the matched global centroid index
        w1_sample_local_indices = [
            i
            for i, v in enumerate(word_sample_to_centroid_indices[w1])
            if v == idx1_global
        ]
        w2_sample_local_indices = [
            i
            for i, v in enumerate(word_sample_to_centroid_indices[w2])
            if v == idx2_global
        ]
        w1_range = word_to_embedding_indices[w1]  # e.g., [0, 100]
        w2_range = word_to_embedding_indices[w2]  # e.g., [100, 200]
        w1_embedding_indices = [w1_range[0] + i for i in w1_sample_local_indices]
        w2_embedding_indices = [w2_range[0] + i for i in w2_sample_local_indices]

        src = torch.as_tensor(
            embeddings_memmap[w1_embedding_indices].copy(), device=device
        )
        tgt = torch.as_tensor(
            embeddings_memmap[w2_embedding_indices].copy(), device=device
        )

        # Z-score normalization
        src = (src - global_embedding_mean) / (global_embedding_std + eps)
        tgt = (tgt - global_embedding_mean) / (global_embedding_std + eps)

        # L2 normalization
        src_normed = torch.nn.functional.normalize(src, p=2, dim=1)
        tgt_normed = torch.nn.functional.normalize(tgt, p=2, dim=1)

        n_src, n_tgt = src.shape[0], tgt.shape[0]
        if n_src == 0 or n_tgt == 0:
            print(f"Skipping {w1}->{w2}: Source or Target cluster is empty.")
            continue
        # Uniform weights
        a = torch.ones(n_src, device=device) / n_src
        b = torch.ones(n_tgt, device=device) / n_tgt

        # Cost matrix: squared Euclidean distance
        M = torch.cdist(src_normed, tgt_normed, p=2) ** 2

        with torch.no_grad():
            # Sinkhorn algorithm
            P = ot.sinkhorn(
                a, b, M, reg=reg, numItermax=max_iter, stopThr=stop_threshold
            )
            if not torch.isfinite(P).all():  # type: ignore
                # skip bad cases
                continue
            # Barycentric projection
            # (n_src, d)
            # division by a is redundant due to normalization, kept for clarity
            dest = torch.matmul(P, tgt_normed) / a[:, None]  # type: ignore
            dest_normed = torch.nn.functional.normalize(dest, p=2, dim=1)

            disp = dest_normed - src_normed

        displacements_memmap[index : index + n_src] = disp.cpu().numpy()
        displacement_items.append(
            {
                "w1": w1,
                "w2": w2,
                "src_idxs": w1_embedding_indices,
                "target_embedding_indices": w2_embedding_indices,
                "tgt_centroid_idx": idx2_global,
                "disp_idxs": list(range(index, index + n_src)),
            }
        )
        index += n_src

    displacements_memmap.flush()

    all_indices = []

    for item in displacement_items:
        for s_idx, d_idx in zip(item["src_idxs"], item["disp_idxs"]):
            all_indices.append([s_idx, d_idx, item["tgt_centroid_idx"]])

    indices_matrix = np.array(all_indices, dtype=np.int32)
    np.save(displacement_indices_path, indices_matrix)

    output = {
        "reg": reg,
        "max_iter": max_iter,
        "stop_threshold": stop_threshold,
        "displacement_count": index,
    }
    with open(displacement_metadata_path, "w") as f:
        json.dump(output, f)
    print(
        f"Saved displacements to {displacements_path}, indices to {displacement_indices_path}, and metadata to {displacement_metadata_path}"
    )


def run_displacement_step(cfg):
    pairs_path = Path(cfg["data"]["dir"]) / cfg["data"]["pairs_path"]
    embeddings_path = Path(cfg["data"]["dir"]) / cfg["data"]["embeddings_path"]
    embedding_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["embedding_metadata_path"]
    )
    centroids_path = Path(cfg["data"]["dir"]) / cfg["data"]["centroids_path"]
    centroid_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
    )
    displacements_path = Path(cfg["data"]["dir"]) / cfg["data"]["displacements_path"]
    displacement_indices_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["displacement_indices_path"]
    )
    displacement_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["displacement_metadata_path"]
    )

    max_sample_count_per_word = cfg["preprocess"]["vocab"]["max_sample_count_per_word"]
    reg = float(cfg["preprocess"]["ot"]["reg"])
    max_iter = cfg["preprocess"]["ot"]["max_iter"]
    stop_threshold = float(cfg["preprocess"]["ot"]["stop_threshold"])

    pairs_df = pd.read_csv(pairs_path, keep_default_na=False, na_values=[])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(embedding_metadata_path, "r") as f:
        embedding_metadata = json.load(f)
    with open(centroid_metadata_path, "r") as f:
        centroid_metadata = json.load(f)
    compute_and_save_displacements(
        pairs_df,
        embeddings_path,
        embedding_metadata,
        centroids_path,
        centroid_metadata,
        displacements_path,
        displacement_indices_path,
        displacement_metadata_path,
        max_sample_count_per_word,
        reg,
        max_iter,
        stop_threshold,
        device,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_displacement_step(cfg)


if __name__ == "__main__":
    main(parse_args())
