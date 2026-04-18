import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def load_results(file_path):
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def summarize_results(results, label):
    reach_target_results = [
        obj for obj in results if obj.get("word_path", None) != None
    ]
    with_inermediate_results = [
        obj
        for obj in reach_target_results
        if isinstance(obj, dict) and obj.get("expansion_count", 0) > 1
    ]

    print(f"{label} results:")

    if not with_inermediate_results:
        print("No results with intermediate words found.\n")
        return

    avg_expansion = np.mean([r["expansion_count"] for r in with_inermediate_results])
    avg_expansion_low_simlex_score = np.mean(
        [r["expansion_count"] for r in reach_target_results if r["simlex_score"] < 3.33]
    )
    avg_expansion_mid_simlex_score = np.mean(
        [
            r["expansion_count"]
            for r in reach_target_results
            if r["simlex_score"] >= 3.33 and r["simlex_score"] < 6.66
        ]
    )
    avg_expansion_high_simlex_score = np.mean(
        [
            r["expansion_count"]
            for r in reach_target_results
            if r["simlex_score"] >= 6.66
        ]
    )
    avg_similarity = np.mean(
        [np.mean(r["cosine_similarity"]) for r in with_inermediate_results]
    )
    avg_path_length = np.mean([len(r["word_path"]) for r in with_inermediate_results])
    print(f"Avg. word cosine similarity within path: {avg_similarity:.6f}")
    print(f"Avg. path length: {avg_path_length:.6f}")
    print(f"Avg. node expansion: {avg_expansion:.6f}")
    print(
        f"Avg. node expansion (words with SimLex999 score 0≤x<3.33): {avg_expansion_low_simlex_score:.6f}"
    )
    print(
        f"Avg. node expansion (words with SimLex999 score 3.33≤x<6.66): {avg_expansion_mid_simlex_score:.6f}"
    )
    print(
        f"Avg. node expansion (words with SimLex999 score 6.66≤x≤10): {avg_expansion_high_simlex_score:.6f}"
    )
    print(
        f"% of paths reaching target: "
        f"{(len(reach_target_results) / len(results) * 100):.6f}"
    )

    print(
        f"% of paths with intermediate words: "
        f"{(len(with_inermediate_results) / len(results) * 100):.6f}"
    )


def analyze(cfg):
    results_dir = Path(cfg["results"]["dir"])

    a_star_flow_results = load_results(results_dir / "results_a_star_flow.jsonl")
    a_star_vanilla_results = load_results(results_dir / "results_a_star_vanilla.jsonl")

    summarize_results(a_star_flow_results, "A* with flow guidance")
    summarize_results(a_star_vanilla_results, "A* vanilla")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    analyze(cfg)


if __name__ == "__main__":
    main(parse_args())
