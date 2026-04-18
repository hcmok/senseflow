import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.utils import slerp


class OTDisplacementDataset(Dataset):
    """
    Returns:
        z_t : torch.Tensor  # interpolated point on geodesic (unit sphere)
        g   : torch.Tensor  # target centroid
        t   : torch.Tensor  # time ∈ [0,1]
        v   : torch.Tensor  # ground-truth velocity at z_t
    """

    def __init__(
        self,
        embeddings_path,
        centroids_path,
        displacements_path,
        displacement_indices_path,
        embedding_metadata,
        centroid_metadata,
        displacement_metadata,
    ):
        d = embedding_metadata["dimension"]
        embedding_count = embedding_metadata["embedding_count"]
        self.global_embedding_mean = np.array(
            embedding_metadata["global_embedding_mean"], dtype=np.float32
        )
        self.global_embedding_std = np.array(
            embedding_metadata["global_embedding_std"], dtype=np.float32
        )

        centroid_count = centroid_metadata["centroid_count"]

        self.displacement_count = displacement_metadata["displacement_count"]
        self.displacement_indices = np.load(displacement_indices_path, mmap_mode="r")

        self.embeddings_memmap = np.memmap(
            embeddings_path, dtype=np.float32, mode="r", shape=(embedding_count, d)
        )
        self.centroids_memmap = np.memmap(
            centroids_path, dtype=np.float32, mode="r", shape=(centroid_count, d)
        )
        self.displacements_memmap = np.memmap(
            displacements_path,
            dtype=np.float32,
            mode="r",
            shape=(self.displacement_count, d),
        )

    def __len__(self):
        return self.displacement_count

    def __getitem__(self, idx):
        emb_idx, disp_idx, tgt_centroid_idx = self.displacement_indices[idx]
        disp = self.displacements_memmap[disp_idx]
        x = self.embeddings_memmap[emb_idx]
        g = self.centroids_memmap[tgt_centroid_idx]

        # Z-score and l2 norm, centroids are already normed
        x = (x - self.global_embedding_mean) / self.global_embedding_std
        x = x / (np.linalg.norm(x) + 1e-8)

        # Project displacement onto the tangent plane at x
        v_tangent = disp - np.dot(disp, x) * x
        v_norm = np.linalg.norm(v_tangent)
        if v_norm < 1e-8:
            y = x
        else:
            # Riemannian exponential map on a unit sphere
            y = np.cos(v_norm) * x + np.sin(v_norm) * (v_tangent / v_norm)

        t = np.random.rand()

        # Spherical linear interpolation
        z_t = slerp(x, y, t)

        dot = np.clip(np.dot(x, y), -1.0, 1.0)
        theta = np.arccos(dot)

        if theta < 1e-8:
            # Handle cases like very small v_norm and y set to x
            v = np.zeros_like(x)
        else:
            # Time derivative of SLERP
            v = (theta / (np.sin(theta) + 1e-8)) * (
                -np.cos((1 - t) * theta) * x + np.cos(t * theta) * y
            )

        z_t = torch.from_numpy(z_t).float()
        g = torch.from_numpy(g).float()
        t = torch.tensor([t]).float()
        v = torch.from_numpy(v).float()

        return z_t, g, t, v
