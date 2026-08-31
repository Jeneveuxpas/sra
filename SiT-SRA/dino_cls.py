import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.pos_embed import resample_abs_pos_embed


class DINOCLSExtractor(nn.Module):
    """Frozen DINOv2 wrapper that returns a clean normalized CLS token."""

    def __init__(self, backbone, resolution):
        super().__init__()
        if resolution not in (256, 512):
            raise ValueError(f"Unsupported image resolution: {resolution}")
        self.backbone = backbone
        self.input_size = 224 * (resolution // 256)
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
        if not isinstance(output, dict) or output.get("x_norm_clstoken") is None:
            raise RuntimeError("DINOv2 did not return x_norm_clstoken")
        cls_condition = output["x_norm_clstoken"]
        if cls_condition.shape != (raw_images.shape[0], self.embed_dim):
            raise RuntimeError(
                f"Unexpected DINO CLS shape {tuple(cls_condition.shape)}; "
                f"expected ({raw_images.shape[0]}, {self.embed_dim})"
            )
        return cls_condition


def load_dino_cls_extractor(model_name, resolution, device, accelerator):
    """Load one frozen DINOv2 copy per rank after warming each node's cache."""
    with accelerator.local_main_process_first():
        backbone = torch.hub.load("facebookresearch/dinov2", model_name)

    patch_grid = 16 * (resolution // 256)
    backbone.pos_embed.data = resample_abs_pos_embed(
        backbone.pos_embed.data, [patch_grid, patch_grid]
    )
    extractor = DINOCLSExtractor(backbone, resolution)
    return extractor.to(device).eval()
