import argparse
import logging
import sys

import yaml

from src.data.centroids import run_centroid_step
from src.data.displacements import run_displacement_step
from src.data.embeddings import run_embedding_step
from src.data.init import run_init_step
from src.data.pairs import run_pair_step
from src.data.samples import run_sample_step
from src.model.train import run_train_step
from src.visualize.semantic_manifold import run_semantic_manifold_step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for name in ["httpx", "urllib3", "requests", "datasets", "huggingface_hub"]:
    logging.getLogger(name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--step", type=str, default="all")

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded config from: {args.config}")

    # Execute based on the selected step
    steps = [
        ("init", run_init_step),
        ("sample", run_sample_step),
        ("embedding", run_embedding_step),
        ("centroid", run_centroid_step),
        ("pair", run_pair_step),
        ("displacement", run_displacement_step),
        ("train", run_train_step),
        ("semantic_manifold", run_semantic_manifold_step),
    ]

    try:
        for step_name, step_fn in steps:
            if args.step in [step_name, "all"]:
                logger.info(f"Starting step: {step_name}")
                step_fn(cfg)

    except Exception as e:
        logger.error(f"Pipeline failed during step '{args.step}': {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main(parse_args())
