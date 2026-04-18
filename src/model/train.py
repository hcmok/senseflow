import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.ot_dataset import OTDisplacementDataset
from src.model.model import WordVectorFieldModel


def load_latest_checkpoint(model, optimizer, checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)

    ckpt_files = list(checkpoint_dir.glob("model_epoch_*.pth"))
    if not ckpt_files:
        print("No checkpoints found.")
        return -1  # start from scratch

    def extract_epoch(path):
        match = re.search(r"model_epoch_(\d+)\.pth", path.name)
        return int(match.group(1)) if match else -1

    latest_ckpt = max(ckpt_files, key=extract_epoch)
    latest_epoch = extract_epoch(latest_ckpt)
    ckpt = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Loaded checkpoint: {latest_ckpt}; resuming from epoch {latest_epoch+1}")

    return latest_epoch


def train(
    model,
    optimizer,
    train_loader,
    val_loader,
    loss_weights,
    num_epochs,
    checkpoint_dir,
    train_log_path,
    device,
):

    model.to(device)

    last_epoch = load_latest_checkpoint(model, optimizer, checkpoint_dir, device)
    start_epoch = last_epoch + 1
    if start_epoch >= num_epochs:
        print(
            f"Training already completed (start_epoch={start_epoch}, num_epochs={num_epochs})"
        )
        return

    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs} [Train]",
            leave=False,
            dynamic_ncols=True,
        )

        train_loss = 0.0
        train_loss_mse = 0.0
        train_loss_ortho = 0.0
        for batch in train_bar:
            batch = [b.to(device) for b in batch]
            z_t, g, t, v_target = batch

            inp = torch.cat([z_t, g, t], dim=-1)
            optimizer.zero_grad()
            v_pred = model(inp)
            mse_loss = F.mse_loss(v_pred, v_target)
            ortho_loss = torch.mean((torch.sum(v_pred * z_t, dim=1)) ** 2)
            loss = loss_weights["mse"] * mse_loss + loss_weights["ortho"] * ortho_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * v_target.size(0)
            train_loss_mse += mse_loss.item() * v_target.size(0)
            train_loss_ortho += ortho_loss.item() * v_target.size(0)
            train_bar.set_postfix(
                loss=loss.item(), mse_loss=mse_loss.item(), ortho_loss=ortho_loss.item()
            )

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_mse = train_loss_mse / len(train_loader.dataset)
        avg_train_ortho = train_loss_ortho / len(train_loader.dataset)

        # Validation
        model.eval()
        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{num_epochs} [Val]",
            leave=False,
            dynamic_ncols=True,
        )
        val_loss = 0.0
        val_loss_mse = 0.0
        val_loss_ortho = 0.0
        with torch.no_grad():
            for batch in val_bar:
                batch = [b.to(device) for b in batch]
                z_t, g, t, v_target = batch

                inp = torch.cat([z_t, g, t], dim=-1)
                v_pred = model(inp)
                mse_loss = F.mse_loss(v_pred, v_target)
                ortho_loss = torch.mean((torch.sum(v_pred * z_t, dim=1)) ** 2)
                loss = (
                    loss_weights["mse"] * mse_loss + loss_weights["ortho"] * ortho_loss
                )

                val_loss += loss.item() * v_target.size(0)
                val_loss_mse += mse_loss.item() * v_target.size(0)
                val_loss_ortho += ortho_loss.item() * v_target.size(0)
                val_bar.set_postfix(
                    loss=loss.item(),
                    mse_loss=mse_loss.item(),
                    ortho_loss=ortho_loss.item(),
                )
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_mse = val_loss_mse / len(val_loader.dataset)
        avg_val_ortho = val_loss_ortho / len(val_loader.dataset)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            checkpoint_dir / f"model_epoch_{epoch}.pth",
        )
        file_exists = train_log_path.exists()

        with open(train_log_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "train_mse",
                        "train_ortho",
                        "val_loss",
                        "val_mse",
                        "val_ortho",
                    ]
                )

            row_to_write = [epoch] + [
                f"{val:.6f}"
                for val in [
                    avg_train_loss,
                    avg_train_mse,
                    avg_train_ortho,
                    avg_val_loss,
                    avg_val_mse,
                    avg_val_ortho,
                ]
            ]
            writer.writerow(row_to_write)
        print(
            f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )


def run_train_step(cfg):
    embeddings_path = Path(cfg["data"]["dir"]) / cfg["data"]["embeddings_path"]
    centroids_path = Path(cfg["data"]["dir"]) / cfg["data"]["centroids_path"]
    displacements_path = Path(cfg["data"]["dir"]) / cfg["data"]["displacements_path"]

    displacement_indices_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["displacement_indices_path"]
    )

    embedding_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["embedding_metadata_path"]
    )
    centroid_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
    )
    displacement_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["displacement_metadata_path"]
    )
    checkpoint_dir = Path(cfg["checkpoints"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["logs"]["dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = log_dir / cfg["logs"]["train_log_path"]

    with open(embedding_metadata_path, "r") as f:
        embedding_metadata = json.load(f)
    with open(centroid_metadata_path, "r") as f:
        centroid_metadata = json.load(f)
    with open(displacement_metadata_path, "r") as f:
        displacement_metadata = json.load(f)

    input_dim = cfg["model"]["input_dim"]
    output_dim = cfg["model"]["output_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]
    num_layers = cfg["model"]["num_layers"]
    dropout = cfg["model"]["dropout"]

    test_size = cfg["training"]["test_size"]
    batch_size = cfg["training"]["batch_size"]
    lr = float(cfg["training"]["lr"])
    num_epochs = cfg["training"]["num_epochs"]
    loss_weights = cfg["training"]["loss_weights"]

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = WordVectorFieldModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = OTDisplacementDataset(
        embeddings_path,
        centroids_path,
        displacements_path,
        displacement_indices_path,
        embedding_metadata,
        centroid_metadata,
        displacement_metadata,
    )

    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=test_size, shuffle=True, random_state=seed
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train(
        model,
        optimizer,
        train_loader,
        val_loader,
        loss_weights,
        num_epochs,
        checkpoint_dir,
        train_log_path,
        device,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_train_step(cfg)


if __name__ == "__main__":
    main(parse_args())
