import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans


def compute_best_centers(
    sample_embeddings,
    max_clusters,
    min_size,
    max_similarity,
    global_embedding_mean,
    global_embedding_std,
    eps=1e-8,
):
    """
    Compute the best centroids for word embeddings with clustering if needed.
    """
    n_samples = sample_embeddings.shape[0]
    cluster_labels = np.zeros(n_samples, dtype=int)
    best_centers = [np.mean(sample_embeddings, axis=0)]

    if n_samples > min_size * 2:
        # Test different k values
        for k_test in range(2, max_clusters + 1):
            km = KMeans(n_clusters=k_test, n_init="auto", random_state=42).fit(
                sample_embeddings
            )

            centers = km.cluster_centers_
            current_labels = km.labels_
            cluster_sizes = np.bincount(current_labels, minlength=k_test)

            if np.any(cluster_sizes < min_size):
                break

            pairwise_cosine_similarities = [
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                for a, b in combinations(centers, 2)
            ]

            if max(pairwise_cosine_similarities) < max_similarity:
                best_centers = centers
                cluster_labels = current_labels
            else:
                break  # Stop increasing k if centers get too crowded

    # Z-score and L2 normalize
    centroids_z = [
        (c - global_embedding_mean) / (global_embedding_std + eps) for c in best_centers
    ]
    centroids = [c / (np.linalg.norm(c) + eps) for c in centroids_z]

    return centroids, cluster_labels


def compute_and_save_centroids(
    embeddings_path,
    embedding_metadata,
    centroids_path,
    centroid_metadata_path,
    max_clusters,
    min_size,
    max_similarity,
):
    d = embedding_metadata["dimension"]
    embedding_count = embedding_metadata["embedding_count"]
    word_to_embedding_indices = embedding_metadata["word_to_embedding_indices"]
    global_embedding_mean = np.array(
        embedding_metadata["global_embedding_mean"], dtype=np.float32
    )
    global_embedding_std = np.array(
        embedding_metadata["global_embedding_std"], dtype=np.float32
    )

    embeddings_memmap = np.memmap(
        embeddings_path, dtype=np.float32, mode="r", shape=(embedding_count, d)
    )

    centroid_count = 0
    index = 0

    word_to_centroid_indices = {}
    word_sample_to_centroid_indices = {}

    for word, (start, end) in word_to_embedding_indices.items():
        sample_embeddings = embeddings_memmap[start:end]
        centroids, labels = compute_best_centers(
            sample_embeddings,
            max_clusters,
            min_size,
            max_similarity,
            global_embedding_mean,
            global_embedding_std,
        )
        centroid_count += len(centroids)

    centroids_memmap = np.memmap(
        centroids_path, dtype=np.float32, mode="w+", shape=(centroid_count, d)
    )
    for word, (start, end) in word_to_embedding_indices.items():
        sample_embeddings = embeddings_memmap[start:end]
        centroids, labels = compute_best_centers(
            sample_embeddings,
            max_clusters,
            min_size,
            max_similarity,
            global_embedding_mean,
            global_embedding_std,
        )

        centroids_memmap[index : index + len(centroids)] = np.array(centroids).astype(
            "float32"
        )
        word_to_centroid_indices[word] = list(range(index, index + len(centroids)))
        word_sample_to_centroid_indices[word] = (index + labels).tolist()
        index += len(centroids)

    centroids_memmap.flush()
    output = {
        "dimension": d,
        "centroid_count": index,
        "average_clusters": index / len(list(word_to_embedding_indices.items())),
        "word_to_centroid_indices": word_to_centroid_indices,
        "word_sample_to_centroid_indices": word_sample_to_centroid_indices,
    }
    with open(centroid_metadata_path, "w") as f:
        json.dump(output, f)
    print(
        f"Saved centroids to {centroids_path} and metadata to {centroid_metadata_path}"
    )


def run_centroid_step(cfg):
    embeddings_path = Path(cfg["data"]["dir"]) / cfg["data"]["embeddings_path"]
    embedding_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["embedding_metadata_path"]
    )
    centroids_path = Path(cfg["data"]["dir"]) / cfg["data"]["centroids_path"]
    centroid_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
    )

    max_clusters = cfg["preprocess"]["centroids"]["max_clusters"]
    min_size = cfg["preprocess"]["centroids"]["min_size"]
    max_similarity = cfg["preprocess"]["centroids"]["max_similarity"]

    with open(embedding_metadata_path, "r") as f:
        embedding_metadata = json.load(f)
    compute_and_save_centroids(
        embeddings_path,
        embedding_metadata,
        centroids_path,
        centroid_metadata_path,
        max_clusters,
        min_size,
        max_similarity,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_centroid_step(cfg)


if __name__ == "__main__":
    main(parse_args())
