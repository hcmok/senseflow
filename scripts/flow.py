import argparse
import json
from pathlib import Path

import torch
import yaml
from nltk.stem import WordNetLemmatizer

from src.pathfind.morph_a_star import morph_a_star_generator
from src.utils.load import load_model_for_inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="Starting word")
    parser.add_argument("--end", type=str, required=True, help="Target word")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-neighbor-distance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--k-cutoff",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=None,
        metavar=("STEP", "LEMMA", "FLOW"),
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--allow-immediate-reach",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_for_inference_path = (
        Path(cfg["checkpoints"]["dir"])
        / cfg["checkpoints"]["checkpoint_for_inference_path"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_for_inference(
        checkpoint_path=checkpoint_for_inference_path,
        cfg=cfg,
        device=device,
    )
    k_neighbors = (
        args.k_neighbors
        if args.k_neighbors is not None
        else cfg["pathfind"]["kdtree"]["k_neighbors"]
    )
    max_neighbor_distance = (
        args.max_neighbor_distance
        if args.max_neighbor_distance is not None
        else cfg["pathfind"]["kdtree"]["max_neighbor_distance"]
    )
    k_cutoff = (
        args.k_cutoff
        if args.k_cutoff is not None
        else cfg["pathfind"]["kdtree"]["k_cutoff"]
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else cfg["pathfind"]["kdtree"]["temperature"]
    )
    weights = (
        args.weights
        if args.weights is not None
        else cfg["pathfind"]["astar"]["weights"]
    )
    max_expansions = (
        args.max_expansions
        if args.max_expansions is not None
        else cfg["pathfind"]["astar"]["max_expansions"]
    )

    allow_immediate_reach = (
        args.allow_immediate_reach
        if args.allow_immediate_reach is not None
        else cfg["pathfind"]["astar"]["allow_immediate_reach"]
    )

    centroids_path = Path(cfg["data"]["dir"]) / cfg["data"]["centroids_path"]
    centroid_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
    )

    with open(centroid_metadata_path, "r") as f:
        centroid_metadata = json.load(f)

    lemmatizer = WordNetLemmatizer()

    gen = morph_a_star_generator(
        word_start=args.start,
        word_end=args.end,
        model=model,
        lemmatizer=lemmatizer,
        k_neighbors=k_neighbors,
        max_neighbor_distance=max_neighbor_distance,
        k_cutoff=k_cutoff,
        temperature=temperature,
        weights=weights,
        max_expansions=max_expansions,
        allow_immediate_reach=allow_immediate_reach,
        centroids_path=centroids_path,
        centroid_metadata=centroid_metadata,
        device=device,
    )
    result = next(gen)
    if result["status"] == 2:
        word_sense_path = result["word_sense_path"]
        word_path = [x.split("_")[0] for x in word_sense_path]  # type: ignore
        dynamic_path = [
            x.split("_")[0] if word_path.count(x.split("_")[0]) == 1 else x
            for x in word_sense_path  # type: ignore
        ]
        print(" → ".join(dynamic_path))


if __name__ == "__main__":
    main(parse_args())
