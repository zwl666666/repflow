"""Compact MMDiT used by the COCO text-to-image experiments."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models import FinalLayer, PatchEmbed, TimestepEmbedder, get_2d_sincos_pos_embed


def attention(q, k, v, heads):
    b, length = q.shape[:2]
    if q.ndim == 4:
        q = q.reshape(b, length, -1)
        k = k.reshape(b, length, -1)
        v = v.reshape(b, length, -1)
    _, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def split_qkv(qkv, heads):
    b, l, three_c = qkv.shape
    c = three_c // 3
    head_dim = c // heads
    qkv = qkv.reshape(b, l, 3, heads, head_dim)
    return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise NotImplementedError(f"qk_norm={qk_norm!r} is not used by this experiment")

    def forward(self, x):
        b, l, c = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.ln_q(q).reshape(b, l, c)
        k = self.ln_k(k).reshape(b, l, c)
        out = attention(q, k, v, self.num_heads)
        return self.proj(out)


class DismantledBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True, pre_only=False):
        super().__init__()
        self.pre_only = pre_only
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads, qkv_bias=qkv_bias)
        if pre_only:
            # The final context stream is pre-only; projection and MLP are never used.
            del self.attn.proj
        else:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            hidden = int(hidden_size * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden, hidden_size),
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, (2 if pre_only else 6) * hidden_size),
        )

    def pre_attention(self, x, c):
        if self.pre_only:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            qkv = self.attn.qkv(modulate(self.norm1(x), shift, scale))
            return qkv
        chunks = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
        qkv = self.attn.qkv(modulate(self.norm1(x), shift_msa, scale_msa))
        return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        x = x + gate_msa.unsqueeze(1) * self.attn.proj(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JointBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True, pre_only=False):
        super().__init__()
        self.context_block = DismantledBlock(
            hidden_size, num_heads, mlp_ratio, qkv_bias, pre_only=pre_only
        )
        self.x_block = DismantledBlock(
            hidden_size, num_heads, mlp_ratio, qkv_bias, pre_only=False
        )

    def forward(self, context, x, c):
        if self.context_block.pre_only:
            context_qkv = self.context_block.pre_attention(context, c)
            context_rest = None
        else:
            context_qkv, context_rest = self.context_block.pre_attention(context, c)
        x_qkv, x_rest = self.x_block.pre_attention(x, c)
        heads = self.x_block.attn.num_heads
        cq, ck, cv = split_qkv(context_qkv, heads)
        xq, xk, xv = split_qkv(x_qkv, heads)
        q = torch.cat((cq, xq), dim=1)
        k = torch.cat((ck, xk), dim=1)
        v = torch.cat((cv, xv), dim=1)
        attn = attention(q, k, v, heads)
        context_len = context_qkv.shape[1]
        context_attn, x_attn = attn[:, :context_len], attn[:, context_len:]
        if self.context_block.pre_only:
            context = None
        else:
            context = self.context_block.post_attention(context_attn, *context_rest)
        x = self.x_block.post_attention(x_attn, *x_rest)
        return context, x


class MMDiT(nn.Module):
    """REPA-style compact MMDiT with explicit RepFlow feature taps."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        depth=24,
        mlp_ratio=4.0,
        context_dim=768,
        qkv_bias=True,
        learn_sigma=False,
        pos_embed_max_size=16,
    ):
        super().__init__()
        hidden_size = 32 * depth
        num_heads = depth
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.pos_embed_max_size = pos_embed_max_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.context_embedder = nn.Linear(context_dim, hidden_size)
        self.joint_blocks = nn.ModuleList(
            [
                JointBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    pre_only=(i == depth - 1),
                )
                for i in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        num_patches = self.x_embedder.num_patches
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, num_patches, hidden_size),
            persistent=False,
        )
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size,
            int(num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def cropped_pos_embed(self, hw):
        p = self.patch_size
        h, w = hw
        hp, wp = h // p, w // p
        if hp > self.pos_embed_max_size or wp > self.pos_embed_max_size:
            raise ValueError(f"Patch grid {(hp, wp)} exceeds max size {self.pos_embed_max_size}")
        top = (self.pos_embed_max_size - hp) // 2
        left = (self.pos_embed_max_size - wp) // 2
        spatial = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial = spatial[:, top : top + hp, left : left + wp, :]
        return rearrange(spatial, "1 h w c -> 1 (h w) c")

    def unpatchify(self, x, hw):
        c = self.out_channels
        p = self.patch_size
        h, w = hw
        hp, wp = h // p, w // p
        x = x.reshape(x.shape[0], hp, wp, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h, w)

    def forward(
        self,
        x,
        t,
        context,
        return_features=False,
        g2r_layers: Optional[Sequence[int]] = None,
        r2g_layer: Optional[int] = None,
    ):
        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(hw)
        c = self.t_embedder(t).to(dtype=x.dtype)
        context = self.context_embedder(context)

        g2r_features: Dict[int, torch.Tensor] = {}
        r2g_feature = None
        requested_g2r = set(g2r_layers or [])
        for i, block in enumerate(self.joint_blocks):
            context, x = block(context, x, c)
            if return_features and i in requested_g2r:
                g2r_features[i] = x.detach()
            if return_features and r2g_layer is not None and i == r2g_layer:
                r2g_feature = x

        x = self.final_layer(x, c)
        x = self.unpatchify(x, hw)
        return x, g2r_features, r2g_feature
