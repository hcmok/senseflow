import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
from tqdm import tqdm

from src.experiment.get_word_pairs import get_stratified_simlex_pairs
from src.utils.utils import slerp


def run_geometric_interpolation(pairs, cfg, method, n_steps, emb_model, results_path):
    centroids_path = Path(cfg["data"]["dir"]) / cfg["data"]["centroids_path"]
    centroid_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
    )

    with open(centroid_metadata_path, "r") as f:
        centroid_metadata = json.load(f)

    centroid_count = centroid_metadata["centroid_count"]
    d = centroid_metadata["dimension"]
    centroids_memmap = np.memmap(
        centroids_path, dtype=np.float32, mode="r", shape=(centroid_count, d)
    )
    word_to_centroid_indices = centroid_metadata["word_to_centroid_indices"]
    kdtree = KDTree(centroids_memmap)

    index_to_word = {}
    for word, indices in word_to_centroid_indices.items():
        for i, idx in enumerate(indices):
            index_to_word[idx] = word + f"_{i}"

    results = []
    for word_start, word_end, simlex_score in tqdm(
        pairs, total=len(pairs), desc="Processing word pairs"
    ):

        idx_a = word_to_centroid_indices.get(f"{word_start}")
        idx_b = word_to_centroid_indices.get(f"{word_end}")
        a_senses = centroids_memmap[idx_a]
        b_senses = centroids_memmap[idx_b]

        # Find the best pair of senses to initialize the search
        best_pair = min(
            [(i, j) for i in range(len(a_senses)) for j in range(len(b_senses))],
            key=lambda ij: np.linalg.norm(a_senses[ij[0]] - b_senses[ij[1]]),
        )

        start = f"{word_start}_{best_pair[0]}"
        end = f"{word_end}_{best_pair[1]}"
        path = [start]
        v0 = a_senses[best_pair[0]]
        v1 = b_senses[best_pair[1]]

        for i in range(1, n_steps + 1):
            t = i / n_steps
            if method == "lerp":
                vt = (1 - t) * v0 + t * v1
                vt = vt / (np.linalg.norm(vt) + 1e-8)
            elif method == "slerp":
                vt = slerp(v0, v1, t)
                vt = vt / (np.linalg.norm(vt) + 1e-8)

            _, idx = kdtree.query(vt.reshape(1, -1), k=1)  # type: ignore
            word = index_to_word[idx[0][0]]
            if word != path[-1]:
                path.append(word)
        if end != path[-1]:
            path.append(end)

        word_sense_path = path
        word_path = [x.split("_")[0] for x in word_sense_path]
        dyanamic_path = [
            x.split("_")[0] if word_path.count(x.split("_")[0]) == 1 else x
            for x in word_sense_path
        ]
        emb = emb_model.encode(word_path)
        similarity_chain = []
        for i in range(len(word_path) - 1):
            sim = cosine_similarity([emb[i]], [emb[i + 1]])[0][0]  # type: ignore
            similarity_chain.append(round(float(sim), 6))
        similarity_chain = [float(x) for x in similarity_chain]

        results.append(
            {
                "word_start": word_start,
                "word_end": word_end,
                "simlex_score": simlex_score,
                "word_path": dyanamic_path,
                "cosine_similarity": similarity_chain,
            }
        )

    with open(results_path, "w", encoding="utf-8") as f:
        for result in results:
            entry = {
                "word_start": result["word_start"],
                "word_end": result["word_end"],
                "simlex_score": result["simlex_score"],
                "word_path": result["word_path"],
                "cosine_similarity": result["cosine_similarity"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_test(cfg):
    word_pairs_path = Path(cfg["experiment"]["word_pairs_path"])
    num_word_pairs = cfg["experiment"]["num_word_pairs"]
    results_dir = Path(cfg["results"]["dir"])

    centroid_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
    )
    with open(centroid_metadata_path, "r") as f:
        centroid_metadata = json.load(f)
    word_to_centroid_indices = centroid_metadata["word_to_centroid_indices"]

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    pairs = get_stratified_simlex_pairs(
        word_pairs_path, num_word_pairs, word_to_centroid_indices
    )

    n_steps = cfg["experiment"]["comparisons"]["geometric_interpolation"]["n_steps"]
    run_geometric_interpolation(
        pairs,
        cfg,
        "lerp",
        n_steps,
        emb_model,
        results_dir / f"results_lerp.jsonl",
    )
    run_geometric_interpolation(
        pairs,
        cfg,
        "slerp",
        n_steps,
        emb_model,
        results_dir / f"results_slerp.jsonl",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_test(cfg)


if __name__ == "__main__":
    main(parse_args())
