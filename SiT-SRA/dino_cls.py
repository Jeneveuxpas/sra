import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.pos_embed import resample_abs_pos_embed


class DINOCLSExtractor(nn.Module):
    """Frozen DINOv2 wrapper returning CLS and spatially pooled patch tokens."""

    def __init__(self, backbone, resolution, pool_size=0, include_cls=True):
        super().__init__()
        if resolution not in (256, 512):
            raise ValueError(f"Unsupported image resolution: {resolution}")
        self.backbone = backbone
        self.input_size = 224 * (resolution // 256)
        self.pool_size = pool_size
        self.include_cls = include_cls
        if pool_size < 0:
            raise ValueError(f"pool_size must be non-negative, got {pool_size}")
        if pool_size == 0 and not include_cls:
            raise ValueError("At least one DINO token is required")
        self.num_condition_tokens = (1 if include_cls else 0) + pool_size ** 2
        self.embed_dim = getattr(backbone, "embed_dim", None)
        if self.embed_dim is None:
            raise ValueError("DINOv2 backbone does not expose embed_dim")
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, raw_images):
        if raw_images.ndim != 4 or raw_images.shape[1] != 3:
            raise ValueError(
                f"Expected raw RGB images with shape [B, 3, H, W], got {tuple(raw_images.shape)}"
            )
        images = raw_images.to(dtype=torch.float32) / 255.0
        images = (images - self.image_mean) / self.image_std
        images = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
        )
        output = self.backbone.forward_features(images)
        if not isinstance(output, dict):
            raise RuntimeError("DINOv2 forward_features must return a dictionary")

        condition_tokens = []
        cls_condition = output.get("x_norm_clstoken")
        if self.include_cls and cls_condition is None:
            raise RuntimeError("DINOv2 did not return x_norm_clstoken")
        if self.include_cls and cls_condition.shape != (raw_images.shape[0], self.embed_dim):
            raise RuntimeError(
                f"Unexpected DINO CLS shape {tuple(cls_condition.shape)}; "
                f"expected ({raw_images.shape[0]}, {self.embed_dim})"
            )
        if self.include_cls:
            condition_tokens.append(cls_condition.unsqueeze(1))

        if self.pool_size > 0:
            patch_tokens = output.get("x_norm_patchtokens")
            if patch_tokens is None or patch_tokens.ndim != 3:
                raise RuntimeError("DINOv2 did not return [B, T, D] x_norm_patchtokens")
            batch_size, num_patches, channels = patch_tokens.shape
            patch_grid = math.isqrt(num_patches)
            if patch_grid * patch_grid != num_patches or channels != self.embed_dim:
                raise RuntimeError(
                    f"Unexpected DINO patch shape {tuple(patch_tokens.shape)}"
                )
            if self.pool_size > patch_grid:
                raise ValueError(
                    f"pool_size={self.pool_size} exceeds DINO patch grid {patch_grid}"
                )
            patch_map = patch_tokens.transpose(1, 2).reshape(
                batch_size, channels, patch_grid, patch_grid
            )
            pooled = F.adaptive_avg_pool2d(
                patch_map, output_size=(self.pool_size, self.pool_size)
            )
            pooled = pooled.flatten(2).transpose(1, 2)
            condition_tokens.append(pooled)

        tokens = torch.cat(condition_tokens, dim=1)
        if tokens.shape != (
            raw_images.shape[0], self.num_condition_tokens, self.embed_dim
        ):
            raise RuntimeError(f"Unexpected pooled DINO shape {tuple(tokens.shape)}")

        # Preserve the original single-CLS interface and checkpoint behavior.
        return tokens[:, 0] if self.num_condition_tokens == 1 else tokens


def load_dino_cls_extractor(
    model_name, resolution, device, accelerator, pool_size=0, include_cls=True
):
    """Load one frozen DINOv2 copy per rank after warming each node's cache."""
    with accelerator.local_main_process_first():
        backbone = torch.hub.load("facebookresearch/dinov2", model_name)

    patch_grid = 16 * (resolution // 256)
    backbone.pos_embed.data = resample_abs_pos_embed(
        backbone.pos_embed.data, [patch_grid, patch_grid]
    )
    extractor = DINOCLSExtractor(
        backbone, resolution, pool_size=pool_size, include_cls=include_cls
    )
    return extractor.to(device).eval()
