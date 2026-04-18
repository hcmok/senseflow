import argparse
import json
from pathlib import Path

import torch
import yaml
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.experiment.get_word_pairs import get_stratified_simlex_pairs
from src.pathfind.morph_a_star import morph_a_star_generator
from src.utils.load import load_model_for_inference


def run_a_star_custom_weights(pairs, cfg, weights, emb_model, results_path):
    centroids_path = Path(cfg["data"]["dir"]) / cfg["data"]["centroids_path"]
    centroid_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
    )
    checkpoint_for_inference_path = (
        Path(cfg["checkpoints"]["dir"])
        / cfg["checkpoints"]["checkpoint_for_inference_path"]
    )

    with open(centroid_metadata_path, "r") as f:
        centroid_metadata = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model_for_inference(
        checkpoint_path=checkpoint_for_inference_path, cfg=cfg, device=device
    )
    lemmatizer = WordNetLemmatizer()

    k_neighbors = cfg["pathfind"]["kdtree"]["k_neighbors"]
    max_neighbor_distance = cfg["pathfind"]["kdtree"]["max_neighbor_distance"]
    k_cutoff = cfg["pathfind"]["kdtree"]["k_cutoff"]
    temperature = cfg["pathfind"]["kdtree"]["temperature"]
    max_expansions = cfg["pathfind"]["astar"]["max_expansions"]
    allow_immediate_reach = cfg["pathfind"]["astar"]["allow_immediate_reach"]

    results = []
    for word_start, word_end, simlex_score in tqdm(
        pairs, total=len(pairs), desc="Processing word pairs"
    ):
        gen = morph_a_star_generator(
            word_start=word_start,
            word_end=word_end,
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
            expansion_count = result["expansion_count"]
            emb = emb_model.encode(word_path)
            similarity_chain = []
            for i in range(len(word_path) - 1):
                sim = cosine_similarity([emb[i]], [emb[i + 1]])[0][0]  # type: ignore
                similarity_chain.append(round(float(sim), 6))
            similarity_chain = [float(x) for x in similarity_chain]
        else:
            dynamic_path = None
            expansion_count = None
            similarity_chain = None

        results.append(
            {
                "word_start": word_start,
                "word_end": word_end,
                "simlex_score": simlex_score,
                "word_path": dynamic_path,
                "expansion_count": expansion_count,
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
                "expansion_count": result["expansion_count"],
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

    a_star_flow_weights = cfg["experiment"]["comparisons"]["a_star_flow"]["weights"]
    a_star_flow_weights = [float(x) for x in a_star_flow_weights]
    a_star_vanilla_weights = cfg["experiment"]["comparisons"]["a_star_vanilla"][
        "weights"
    ]
    a_star_vanilla_weights = [float(x) for x in a_star_vanilla_weights]
    run_a_star_custom_weights(
        pairs,
        cfg,
        a_star_flow_weights,
        emb_model,
        results_dir / f"results_a_star_flow.jsonl",
    )
    run_a_star_custom_weights(
        pairs,
        cfg,
        a_star_vanilla_weights,
        emb_model,
        results_dir / f"results_a_star_vanilla.jsonl",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_test(cfg)


if __name__ == "__main__":
    main(parse_args())
