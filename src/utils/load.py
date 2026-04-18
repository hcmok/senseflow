import torch

from src.model.model import WordVectorFieldModel


def load_model_for_inference(checkpoint_path, cfg, device):
    input_dim = cfg["model"]["input_dim"]
    output_dim = cfg["model"]["output_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]
    num_layers = cfg["model"]["num_layers"]
    dropout = cfg["model"]["dropout"]

    model = WordVectorFieldModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
