import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None


class HFImgLatentDataset(Dataset):
    """Paired ImageNet images and VAE moments stored as HuggingFace datasets."""

    PRECOMPUTED = {"sdvae-ft-mse-f8d4"}

    def __init__(self, vae_name, data_dir="/dev/shm/data", split="train"):
        if load_from_disk is None:
            raise ImportError(
                "HFImgLatentDataset requires the 'datasets' package; "
                "install the repository requirements first"
            )
        if vae_name not in self.PRECOMPUTED:
            raise ValueError(f"VAE {vae_name} not found in {sorted(self.PRECOMPUTED)}")

        split_dir = "val" if split == "val" else ""
        image_path = os.path.join(data_dir, "imagenet-latents-images", split_dir)
        latent_path = os.path.join(
            data_dir, f"imagenet-latents-{vae_name}", split_dir
        )
        self.img_dataset = load_from_disk(image_path)
        self.latent_dataset = load_from_disk(latent_path)

        if len(self.img_dataset) != len(self.latent_dataset):
            raise ValueError(
                "Image and latent dataset lengths differ: "
                f"images={len(self.img_dataset)}, latents={len(self.latent_dataset)}"
            )
        self._require_columns(self.img_dataset, {"image", "label"}, image_path)
        self._require_columns(self.latent_dataset, {"data"}, latent_path)

    @staticmethod
    def _require_columns(dataset, required, path):
        columns = set(dataset.column_names)
        missing = required - columns
        if missing:
            raise KeyError(f"Dataset {path} is missing columns: {sorted(missing)}")

    def __getitem__(self, idx):
        image_element = self.img_dataset[idx]
        latent_element = self.latent_dataset[idx]
        image = np.asarray(image_element["image"].convert("RGB"), dtype=np.uint8)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        latent = torch.as_tensor(latent_element["data"])
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        if latent.ndim != 3 or latent.shape[0] != 8:
            raise ValueError(
                "Expected VAE moments with shape [8, H, W] or [1, 8, H, W], "
                f"got {tuple(latent.shape)} at index {idx}"
            )
        label = image_element["label"]
        return torch.from_numpy(image), latent, torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)

    def __repr__(self):
        return f"HFImgLatentDataset(n={len(self)})"
