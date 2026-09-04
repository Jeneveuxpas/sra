# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from torch.nn.init import trunc_normal_


class SimpleHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleHead, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim+out_dim)
        self.linear2 = nn.Linear(in_dim+out_dim, out_dim)
        self.act = nn.SiLU()
    def forward(self, x):
        x=self.linear1(x)
        x=self.linear2(self.act(x))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        embeddings = self.embedding_table(labels)
        return embeddings, labels


class CLSTokenEmbedder(nn.Module):
    """Projects DINO features and adds fixed positions to privileged tokens."""

    def __init__(
            self,
            input_dim,
            hidden_size,
            num_tokens=1,
            has_cls_token=True,
            target_grid_size=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.has_cls_token = has_cls_token
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        self.projection = nn.Linear(input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        # Match REG for the global CLS slot (zero position). Pooled DINO patch
        # positions are expressed in the same coordinate system as the SiT
        # latent-token grid. A pooled cell is placed at the center of the SiT
        # cells it covers, following the scaled-reference idea used by ControlSFT.
        num_spatial_tokens = num_tokens - int(has_cls_token)
        if num_spatial_tokens:
            grid_size = int(num_spatial_tokens ** 0.5)
            if grid_size * grid_size != num_spatial_tokens:
                raise ValueError(
                    "The number of pooled DINO patch tokens must form a square "
                    f"grid, got {num_spatial_tokens}"
                )
            if target_grid_size is None:
                target_grid_size = grid_size
            if target_grid_size <= 0:
                raise ValueError(
                    f"target_grid_size must be positive, got {target_grid_size}"
                )
            scale = target_grid_size / grid_size
            coords = (np.arange(grid_size, dtype=np.float32) + 0.5) * scale - 0.5
            grid = np.meshgrid(coords, coords)
            grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
            spatial_pos = get_2d_sincos_pos_embed_from_grid(hidden_size, grid)
            if has_cls_token:
                privileged_pos = np.concatenate(
                    [np.zeros((1, hidden_size), dtype=np.float32), spatial_pos], axis=0
                )
            else:
                privileged_pos = spatial_pos
        else:
            privileged_pos = np.zeros((1, hidden_size), dtype=np.float32)
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(privileged_pos).float().unsqueeze(0),
            persistent=False,
        )

    def forward(self, cls_condition, batch_size, device, dtype):
        """Return a projected CLS token, or a masked-safe placeholder for sampling."""
        if cls_condition is None:
            # Training-preview sampling has no clean source image.  SiT.forward
            # pairs this placeholder with cls_present=False, so it is never a
            # visible attention key/value and cannot affect image tokens.
            shape = (batch_size, self.projection.out_features)
            if self.num_tokens > 1:
                shape = (batch_size, self.num_tokens, self.projection.out_features)
            return torch.zeros(*shape, device=device, dtype=dtype)

        if cls_condition.ndim == 3 and cls_condition.shape[1] == 1 and self.num_tokens == 1:
            cls_condition = cls_condition[:, 0]
        expected_shape = (
            (batch_size, self.input_dim)
            if self.num_tokens == 1
            else (batch_size, self.num_tokens, self.input_dim)
        )
        if cls_condition.ndim not in (2, 3):
            raise ValueError(
                "cls_condition must contain a batch of DINO feature tokens, "
                f"got {tuple(cls_condition.shape)}"
            )
        if tuple(cls_condition.shape) != expected_shape:
            raise ValueError(
                f"cls_condition must have shape {expected_shape}, "
                f"got {tuple(cls_condition.shape)}"
            )

        cls_condition = cls_condition.to(device=device, dtype=dtype)
        tokens = self.norm(self.projection(cls_condition))
        if tokens.ndim == 2:
            return tokens + self.pos_embed[:, 0].to(device=device, dtype=dtype)
        return tokens + self.pos_embed.to(device=device, dtype=dtype)


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class MaskedAttention(nn.Module):
    """SiT self-attention with a broadcastable per-sample key/value mask."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=False, fused_attn=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_mask=None):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=0.0)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if key_mask is not None:
                attn = attn + key_mask
            x = attn.softmax(dim=-1) @ v
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj(x)


class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_token_mask=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.use_token_mask = use_token_mask
        if use_token_mask:
            self.attn = MaskedAttention(
                hidden_size, num_heads=num_heads, qkv_bias=True,
                qk_norm=block_kwargs["qk_norm"],
                fused_attn=block_kwargs.get("fused_attn", True),
            )
        else:
            self.attn = Attention(
                hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
            if "fused_attn" in block_kwargs:
                self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, key_mask=None, prefix_c=None, num_prefix_tokens=0):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        if prefix_c is None:
            attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
            gate_msa_tokens = gate_msa.unsqueeze(1)
            gate_mlp_tokens = gate_mlp.unsqueeze(1)
            prefix_modulation = None
        else:
            if not 0 < num_prefix_tokens < x.shape[1]:
                raise ValueError(
                    "num_prefix_tokens must split a non-empty prefix from image tokens, "
                    f"got {num_prefix_tokens} for sequence length {x.shape[1]}"
                )
            prefix_modulation = self.adaLN_modulation(prefix_c).chunk(6, dim=-1)
            (
                prefix_shift_msa,
                prefix_scale_msa,
                prefix_gate_msa,
                _,
                _,
                prefix_gate_mlp,
            ) = prefix_modulation
            normed = self.norm1(x)
            attn_input = torch.cat(
                [
                    modulate(
                        normed[:, :num_prefix_tokens],
                        prefix_shift_msa,
                        prefix_scale_msa,
                    ),
                    modulate(
                        normed[:, num_prefix_tokens:], shift_msa, scale_msa
                    ),
                ],
                dim=1,
            )
            gate_msa_tokens = torch.cat(
                [
                    prefix_gate_msa.unsqueeze(1).expand(-1, num_prefix_tokens, -1),
                    gate_msa.unsqueeze(1).expand(-1, x.shape[1] - num_prefix_tokens, -1),
                ],
                dim=1,
            )
            gate_mlp_tokens = torch.cat(
                [
                    prefix_gate_mlp.unsqueeze(1).expand(-1, num_prefix_tokens, -1),
                    gate_mlp.unsqueeze(1).expand(-1, x.shape[1] - num_prefix_tokens, -1),
                ],
                dim=1,
            )
        if self.use_token_mask:
            attn_output = self.attn(attn_input, key_mask=key_mask)
        else:
            attn_output = self.attn(attn_input)
        x = x + gate_msa_tokens * attn_output

        normed = self.norm2(x)
        if prefix_modulation is None:
            mlp_input = modulate(normed, shift_mlp, scale_mlp)
        else:
            prefix_shift_mlp, prefix_scale_mlp = prefix_modulation[3:5]
            mlp_input = torch.cat(
                [
                    modulate(
                        normed[:, :num_prefix_tokens],
                        prefix_shift_mlp,
                        prefix_scale_mlp,
                    ),
                    modulate(
                        normed[:, num_prefix_tokens:], shift_mlp, scale_mlp
                    ),
                ],
                dim=1,
            )
        x = x + gate_mlp_tokens * self.mlp(mlp_input)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            path_type='edm',
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            decoder_hidden_size=768,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            use_cfg=False,
            cls_condition_dim=0,
            cls_condition_num_tokens=1,
            cls_condition_has_cls=True,
            cls_condition_clean_timestep=False,
            **block_kwargs  # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.cls_condition_dim = cls_condition_dim
        self.cls_condition_num_tokens = cls_condition_num_tokens
        self.cls_condition_has_cls = cls_condition_has_cls
        self.cls_condition_clean_timestep = cls_condition_clean_timestep

        if cls_condition_dim < 0:
            raise ValueError(f"cls_condition_dim must be non-negative, got {cls_condition_dim}")
        if cls_condition_num_tokens <= 0:
            raise ValueError(
                f"cls_condition_num_tokens must be positive, got {cls_condition_num_tokens}"
            )


        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        num_patches = self.x_embedder.num_patches
        image_token_grid_size = int(num_patches ** 0.5)
        if image_token_grid_size * image_token_grid_size != num_patches:
            raise ValueError(f"SiT image-token grid must be square, got {num_patches} tokens")
        self.t_embedder = TimestepEmbedder(hidden_size)  # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.cls_token_embedder = (
            CLSTokenEmbedder(
                cls_condition_dim,
                hidden_size,
                num_tokens=cls_condition_num_tokens,
                has_cls_token=cls_condition_has_cls,
                target_grid_size=image_token_grid_size,
            )
            if cls_condition_dim > 0 else None
        )
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                use_token_mask=cls_condition_dim > 0, **block_kwargs
            ) for _ in range(depth)
        ])
        self.ap_head = SimpleHead(hidden_size, hidden_size)
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, ad=4, cls_condition=None, cls_present=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        ad: number of layers to use for self-alignment
        cls_condition: optional clean DINO features with shape (N, D) for the
            legacy CLS-only path or (N, K, D) for pooled spatial tokens
        cls_present: optional per-sample bool mask controlling whether image
            patches may attend to the privileged prefix tokens during training.
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # timestep and class embedding
        t_embed = self.t_embedder(t)  # (N, D)
        y, labels_train = self.y_embedder(y, self.training)  # (N, D)
        c = t_embed + y  # (N, D)
        key_mask = None
        prefix_c = None
        num_privileged_tokens = 0
        if self.cls_token_embedder is not None:
            # Optionally treat DINO as a clean condition at the t=0 endpoint;
            # image tokens always continue to use their sampled diffusion time.
            prefix_c = (
                self.t_embedder(torch.zeros_like(t)) + y
                if self.cls_condition_clean_timestep else c
            )
            cls_is_missing = cls_condition is None
            cls_tokens = self.cls_token_embedder(
                cls_condition,
                batch_size=x.shape[0],
                device=c.device,
                dtype=c.dtype,
            )
            if cls_tokens.ndim == 2:
                cls_tokens = cls_tokens.unsqueeze(1)
            num_privileged_tokens = cls_tokens.shape[1]
            x = torch.cat([cls_tokens, x], dim=1)
            if cls_present is None:
                # No source image means no CLS information.  Keep its reserved
                # token slot fully invisible for training-preview sampling.
                cls_present = torch.zeros(
                    x.shape[0], device=x.device, dtype=torch.bool
                ) if cls_is_missing else torch.ones(
                    x.shape[0], device=x.device, dtype=torch.bool
                )
            elif cls_present.ndim != 1 or cls_present.shape[0] != x.shape[0]:
                raise ValueError(
                    f"cls_present must have shape [{x.shape[0]}], got {tuple(cls_present.shape)}"
                )
            else:
                cls_present = cls_present.to(device=x.device, dtype=torch.bool)

            if cls_is_missing and cls_present.any():
                raise ValueError("cls_present cannot be true when cls_condition is None")

            # Shape [B, 1, 1, T] broadcasts across attention heads and query
            # tokens. For CLS-off samples, no token can read any privileged
            # prefix key/value.
            key_mask = torch.zeros(x.shape[0], 1, 1, x.shape[1], device=x.device, dtype=x.dtype)
            key_mask[~cls_present, :, :, :num_privileged_tokens] = float("-inf")
        elif cls_condition is not None or cls_present is not None:
            raise ValueError("CLS conditioning is disabled because cls_condition_dim is 0")

        for i, block in enumerate(self.blocks):
            x = block(
                x,
                c,
                key_mask=key_mask,
                prefix_c=prefix_c,
                num_prefix_tokens=num_privileged_tokens,
            )
            if (i + 1) == ad:
                # SRA aligns image representations only; privileged prefix
                # tokens participate in attention but are never alignment targets.
                image_tokens = (
                    x[:, self.cls_condition_num_tokens:]
                    if self.cls_token_embedder is not None else x
                )
                if self.training:
                    xr = self.ap_head(image_tokens)
                else:
                    xr = image_tokens
        if self.cls_token_embedder is not None:
            x = x[:, self.cls_condition_num_tokens:]
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x, xr, labels_train


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2, 'SiT-XL/4': SiT_XL_4, 'SiT-XL/8': SiT_XL_8,
    'SiT-L/2': SiT_L_2, 'SiT-L/4': SiT_L_4, 'SiT-L/8': SiT_L_8,
    'SiT-B/2': SiT_B_2, 'SiT-B/4': SiT_B_4, 'SiT-B/8': SiT_B_8,
    'SiT-S/2': SiT_S_2, 'SiT-S/4': SiT_S_4, 'SiT-S/8': SiT_S_8,
}
