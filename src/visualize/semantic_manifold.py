import argparse
import json
from pathlib import Path

import numpy as np
import umap
import yaml
from scipy.stats import gaussian_kde


def run_semantic_manifold_step(cfg):
    centroids_path = Path(cfg["data"]["dir"]) / cfg["data"]["centroids_path"]
    centroid_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
    )
    semantic_manifold_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["semantic_manifold_path"]
    )
    n_neighbors = cfg["visualize"]["umap"]["n_neighbors"]
    min_dist = cfg["visualize"]["umap"]["min_dist"]
    with open(centroid_metadata_path, "r") as f:
        centroid_metadata = json.load(f)
    d = centroid_metadata["dimension"]
    centroid_count = centroid_metadata["centroid_count"]

    centroids_memmap = np.memmap(
        centroids_path, dtype=np.float32, mode="r", shape=(centroid_count, d)
    )

    # Use cosine similarity to capture semantic relationships and map them to Euclidean distance
    reducer = umap.UMAP(
        n_components=3,
        metric="cosine",
        output_metric="euclidean",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )

    coords_3d = reducer.fit_transform(centroids_memmap)

    # Normalize to a unit sphere
    coords_3d = np.asarray(coords_3d, dtype=float)

    center = np.mean(coords_3d, axis=0)
    centered_coords = coords_3d - center

    norms = np.linalg.norm(centered_coords, axis=1, keepdims=True)
    coords_3d_sphere = centered_coords / (norms + 1e-8)

    xy = coords_3d_sphere.T
    density = gaussian_kde(xy)(xy)

    np.savez_compressed(
        semantic_manifold_path,
        coords_3d_sphere=coords_3d_sphere.astype(np.float32),
        density=density.astype(np.float32),
    )

    print(f"Saved semantic manifold data to {semantic_manifold_path}")


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

    run_semantic_manifold_step(cfg)


if __name__ == "__main__":
    main(parse_args())
