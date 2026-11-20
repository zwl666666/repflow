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
from typing import List, Optional, Sequence, Set
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SimpleHead(nn.Module):
    """SRA alignment head (SiT-SRA): 2-layer SiLU MLP, hidden = in_dim + out_dim."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim + out_dim)
        self.linear2 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


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
    def timestep_embedding(t, dim, max_period=10000):
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
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
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    @torch.compile
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
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
    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Fixed sin-cos embedding (buffer: not a trainable Parameter, avoids DDP "no grad" on frozen weights).
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def encode_tokens(
        self,
        x,
        t,
        y,
        use_dropout=None,
        feature_layers: Optional[Sequence[int]] = None,
        align_feature_layer: Optional[int] = None,
        repa_feature_layer: Optional[int] = None,
    ):
        """
        Encode latent inputs into token features before the final prediction head.

        If ``feature_layers`` is a non-empty sequence of block indices (0-based, after each
        ``SiTBlock``), each selected hidden state is layer-normed and concatenated on the last
        dimension in the order given (negative indices follow Python semantics).

        If ``align_feature_layer`` is set, also returns LN token features at that block index
        in the same forward; it need not appear in ``feature_layers``.

        If ``repa_feature_layer`` is set, also returns tokens at that block for REPA.
        """
        x = self.x_embedder(x) + self.pos_embed
        # t = self.t_embedder(t)
        if use_dropout is None:
            use_dropout = self.training
        y = self.y_embedder(y, use_dropout)
        c = y
        feature = None
        align_feature = None
        repa_feature = None
        num_blocks = len(self.blocks)
        ordered_resolved: Optional[List[int]] = None
        capture_blocks: Optional[Set[int]] = None
        if feature_layers is not None and len(feature_layers) > 0:
            ordered_resolved = []
            for fl in feature_layers:
                layer_idx = fl
                if layer_idx < 0:
                    layer_idx = num_blocks + layer_idx
                if not (0 <= layer_idx < num_blocks):
                    raise IndexError(
                        f"feature layer index {fl} resolves to {layer_idx}, "
                        f"out of range for SiT depth {num_blocks}"
                    )
                ordered_resolved.append(layer_idx)
            capture_blocks = set(ordered_resolved)

        align_resolved = None
        if align_feature_layer is not None:
            li = int(align_feature_layer)
            if li < 0:
                li = num_blocks + li
            if not (0 <= li < num_blocks):
                raise IndexError(
                    f"align_feature_layer {align_feature_layer} resolves to {li}, "
                    f"out of range for SiT depth {num_blocks}"
                )
            align_resolved = li

        repa_resolved = None
        if repa_feature_layer is not None:
            ri = int(repa_feature_layer)
            if ri < 0:
                ri = num_blocks + ri
            if not (0 <= ri < num_blocks):
                raise IndexError(
                    f"repa_feature_layer {repa_feature_layer} resolves to {ri}, "
                    f"out of range for SiT depth {num_blocks}"
                )
            repa_resolved = ri

        layer_feats = {}
        for block_idx, block in enumerate(self.blocks):
            x = block(x, c)
            need_ln = (capture_blocks is not None and block_idx in capture_blocks) or (
                align_resolved is not None and block_idx == align_resolved
            ) or (repa_resolved is not None and block_idx == repa_resolved)
            if need_ln:
                ln_x = x#F.layer_norm(x, (x.shape[-1],))
                if capture_blocks is not None and block_idx in capture_blocks:
                    layer_feats[block_idx] = ln_x
                if align_resolved is not None and block_idx == align_resolved:
                    align_feature = ln_x
                if repa_resolved is not None and block_idx == repa_resolved:
                    repa_feature = ln_x

        if ordered_resolved is not None:
            feature = torch.cat([layer_feats[j] for j in ordered_resolved], dim=-1)
            feature = F.layer_norm(feature, (feature.shape[-1],))

        return x, c, feature, align_feature, repa_feature

    def forward_features(self, x, t, y, feature_layer=-1, feature_layers: Optional[Sequence[int]] = None, use_dropout=None):
        """
        Return normalized token features from one or more intermediate SiT blocks (concat on dim=-1).
        """
        layers = feature_layers if feature_layers is not None else [feature_layer]
        _, _, feature, _, _ = self.encode_tokens(
            x, t, y, use_dropout=use_dropout, feature_layers=layers
        )
        return feature

    def forward(
        self,
        x,
        t,
        y,
        return_features=False,
        feature_layer=-1,
        feature_layers=None,
        align_feature_layer: Optional[int] = None,
        repa_feature_layer: Optional[int] = None,
        use_dropout=None,
    ):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        When return_features=True, returns (pred, feature, align_feature, repa_feature);
        align/repa features may be None.
        """
        layers_kw = None
        align_kw = None
        repa_kw = None
        if return_features:
            # Explicit feature_layers (including []) wins; otherwise fall back to [feature_layer].
            layers_kw = feature_layers if feature_layers is not None else [feature_layer]
            align_kw = align_feature_layer
            repa_kw = repa_feature_layer
        token_features, c, feature, align_feat, repa_feat = self.encode_tokens(
            x,
            t,
            y,
            use_dropout=use_dropout,
            feature_layers=layers_kw,
            align_feature_layer=align_kw,
            repa_feature_layer=repa_kw,
        )
        x = self.final_layer(token_features, c)    # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                     # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        if return_features:
            return x, feature, align_feat, repa_feat
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class StudentBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
    @torch.compile
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LatentStudentNet(nn.Module):
    """
    Student network for latent-space representation learning.
    """
    def __init__(
        self,
        input_size,
        patch_size,
        in_channels,
        hidden_size,
        depth,
        num_heads=None,
        mlp_ratio=4.0,
        mask_ratio=0.75,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = max(1, hidden_size // 64)
        assert hidden_size % num_heads == 0, "student hidden_size must be divisible by num_heads"

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.mask_ratio = mask_ratio
        num_patches = self.x_embedder.num_patches
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        self.blocks = nn.ModuleList([
            StudentBlock(hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

    def random_masking(self, x):
        batch_size, num_tokens, dim = x.shape
        num_keep = max(1, int(round(num_tokens * (1.0 - self.mask_ratio))))
        scores = torch.rand(batch_size, num_tokens, device=x.device)
        keep_idx = torch.topk(scores, k=num_keep, dim=1, largest=False).indices
        mask = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        mask.scatter_(1, keep_idx, False)
        x = x.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, dim))
        return x, mask, keep_idx

    def _resolve_feature_layers(self, feature_layers: Sequence[int]) -> List[int]:
        num_blocks = len(self.blocks)
        resolved = []
        for fl in feature_layers:
            layer_idx = int(fl)
            if layer_idx < 0:
                layer_idx = num_blocks + layer_idx
            if not (0 <= layer_idx < num_blocks):
                raise IndexError(
                    f"student feature layer index {fl} resolves to {layer_idx}, "
                    f"out of range for student depth {num_blocks}"
                )
            resolved.append(layer_idx)
        return resolved

    def forward(self, x, use_mask=True, return_mask=False, feature_layers: Optional[Sequence[int]] = None):
        """
        If ``feature_layers`` is a non-empty sequence of block indices (0-based, after each
        ``StudentBlock``), selected hidden states are concatenated on the last (channel) dim.
        Masking (when enabled) is applied before blocks, so all layers share the same tokens.
        The final LayerNorm is applied to the last-block output when ``use_mask`` is True,
        matching the single-layer path.
        """
        x = self.x_embedder(x) + self.pos_embed
        mask = None
        keep_idx = None
        if use_mask and self.mask_ratio > 0:
            x, mask, keep_idx = self.random_masking(x)

        ordered = None
        capture_blocks = None
        if feature_layers is not None and len(feature_layers) > 0:
            ordered = self._resolve_feature_layers(feature_layers)
            capture_blocks = set(ordered)

        last_idx = len(self.blocks) - 1
        layer_feats = {}
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if use_mask and block_idx == last_idx:
                x = self.norm(x)
            if capture_blocks is not None and block_idx in capture_blocks:
                layer_feats[block_idx] = x

        if ordered is not None:
            x = torch.cat([layer_feats[j] for j in ordered], dim=-1)

        if return_mask:
            return x, mask, keep_idx
        return x


class LatentProjector(nn.Module):
    """
    Restore masked student tokens and project them to teacher feature space.
    """
    def __init__(
        self,
        input_size,
        patch_size,
        student_hidden_size,
        teacher_hidden_size,
        projector_hidden_size=384,
        depth=2,
        num_heads=None,
        mlp_ratio=4.0,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = max(1, projector_hidden_size // 64)
        assert projector_hidden_size % num_heads == 0, "projector hidden_size must be divisible by projector num_heads"

        self.num_patches = (input_size // patch_size) ** 2
        self.input_proj = nn.Linear(student_hidden_size, projector_hidden_size)
        self.t_embedder = TimestepEmbedder(projector_hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, projector_hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            StudentBlock(projector_hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(teacher_hidden_size, eps=1e-6)
        self.output_proj = nn.Linear(projector_hidden_size, teacher_hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def restore_masked_tokens(self, x, keep_idx, t):
        batch_size, _, dim = x.shape
        cond_tokens = self.t_embedder(t).to(x.dtype).unsqueeze(1).expand(batch_size, self.num_patches, dim).clone()
        if keep_idx is None:
            return cond_tokens + x
        cond_tokens.scatter_(1, keep_idx.unsqueeze(-1).expand(-1, -1, dim), x)
        return cond_tokens

    def forward(self, x, t, keep_idx=None):
        x = self.input_proj(x)
        if keep_idx is None:
            x = x + self.pos_embed
        else:
            x = self.restore_masked_tokens(x, keep_idx, t)
            x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        x = self.norm(x)
        return x


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

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
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
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}
