"""REPA (DINOv2) helpers: encoder load, preprocess, MLP projector, neg-cosine loss.

All paths are gated by callers via ``repa_loss_weight > 0`` so existing configs
that never touch these APIs stay unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet defaults used by DINOv2 / timm
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DINOV2_EMBED_DIM = {
    "s": 384,
    "b": 768,
    "l": 1024,
    "g": 1536,
}


def parse_dinov2_enc_type(enc_type: str) -> Tuple[str, str, str]:
    """Parse ``dinov2-vit-b`` / ``dinov2_reg-vit-l`` style names."""
    parts = enc_type.replace("_", "-").split("-")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid REPA enc_type={enc_type!r}; expected like 'dinov2-vit-b'."
        )
    encoder_type, architecture, model_config = parts[0], parts[1], parts[2]
    return encoder_type, architecture, model_config.lower()


def dinov2_embed_dim(enc_type: str) -> int:
    _, _, model_config = parse_dinov2_enc_type(enc_type)
    if model_config not in DINOV2_EMBED_DIM:
        raise ValueError(
            f"Unknown DINOv2 size {model_config!r} in enc_type={enc_type!r}; "
            f"expected one of {sorted(DINOV2_EMBED_DIM)}."
        )
    return DINOV2_EMBED_DIM[model_config]


def build_repa_mlp(hidden_size: int, projector_dim: int, z_dim: int) -> nn.Sequential:
    """3-layer MLP projector matching REPA-main ``build_mlp``."""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


TIMM_DINOV2_NAME = {
    "s": "vit_small_patch14_dinov2.lvd142m",
    "b": "vit_base_patch14_dinov2.lvd142m",
    "l": "vit_large_patch14_dinov2.lvd142m",
    "g": "vit_giant_patch14_dinov2.lvd142m",
}


def _load_state_dict_file(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(state, dict) and "teacher" in state:
        state = state["teacher"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported DINOv2 checkpoint type: {type(state)}")
    return state


def _load_dinov2_via_timm(enc_type: str, resolution: int, checkpoint_path: str):
    """Offline load from local hf_timm / pytorch_model.bin (no torch.hub / GitHub)."""
    import timm
    from timm.layers.pos_embed import resample_abs_pos_embed

    _, _, model_config = parse_dinov2_enc_type(enc_type)
    if model_config not in TIMM_DINOV2_NAME:
        raise ValueError(f"No timm DINOv2 mapping for size={model_config!r}")

    img_size = 224 * (resolution // 256)
    patch_resolution = 16 * (resolution // 256)
    encoder = timm.create_model(
        TIMM_DINOV2_NAME[model_config],
        pretrained=False,
        num_classes=0,
        img_size=img_size,
    )
    state = dict(_load_state_dict_file(checkpoint_path))
    pos = state.pop("pos_embed", None)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "pos_embed"]
    if missing:
        print(f"[repa] timm DINOv2 missing keys (first 8): {missing[:8]}")
    if unexpected:
        print(f"[repa] timm DINOv2 unexpected keys (first 8): {unexpected[:8]}")
    if pos is not None:
        encoder.pos_embed.data = resample_abs_pos_embed(pos, [patch_resolution, patch_resolution])
    else:
        encoder.pos_embed.data = resample_abs_pos_embed(
            encoder.pos_embed.data, [patch_resolution, patch_resolution]
        )
    print(f"[repa] Loaded DINOv2 via timm from {checkpoint_path}")
    return encoder


def load_dinov2_encoder(
    enc_type: str,
    device: torch.device,
    resolution: int = 256,
    repo_or_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
):
    """
    Load a frozen DINOv2 ViT and resample absolute pos embed to match
    ``(resolution // 16)^2`` patch tokens (same as REPA-main).

    ``repo_or_dir``: local clone of facebookresearch/dinov2 for ``torch.hub.load``.
    ``checkpoint_path``: local weights. When set and ``repo_or_dir`` is empty,
    load via timm offline (for hf_timm ``pytorch_model.bin``); otherwise use hub.
    """
    if resolution not in (256, 512):
        raise ValueError(f"REPA currently supports resolution 256/512, got {resolution}")

    encoder_type, architecture, model_config = parse_dinov2_enc_type(enc_type)
    if "dinov2" not in encoder_type:
        raise NotImplementedError(
            f"Only DINOv2 encoders are supported for now, got enc_type={enc_type!r}"
        )
    if architecture != "vit":
        raise NotImplementedError(f"Expected vit architecture, got {architecture!r}")

    # Prefer fully offline timm path when only a local weight file is given.
    if checkpoint_path and not repo_or_dir:
        encoder = _load_dinov2_via_timm(enc_type, resolution, checkpoint_path)
    else:
        hub_entry = (
            f"dinov2_vit{model_config}14_reg"
            if "reg" in encoder_type
            else f"dinov2_vit{model_config}14"
        )
        hub_source = repo_or_dir if repo_or_dir else "facebookresearch/dinov2"
        source_kind = "local" if repo_or_dir else "github"
        encoder = torch.hub.load(
            hub_source, hub_entry, source=source_kind, pretrained=not bool(checkpoint_path)
        )
        if checkpoint_path:
            state = _load_state_dict_file(checkpoint_path)
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            if missing:
                print(f"[repa] DINOv2 load missing keys (first 8): {missing[:8]}")
            if unexpected:
                print(f"[repa] DINOv2 load unexpected keys (first 8): {unexpected[:8]}")

        if hasattr(encoder, "head"):
            encoder.head = nn.Identity()

        import timm

        patch_resolution = 16 * (resolution // 256)
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data,
            [patch_resolution, patch_resolution],
        )

    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def preprocess_for_dinov2(
    x: torch.Tensor,
    resolution: Optional[int] = None,
    input_range: str = "01",
) -> torch.Tensor:
    """
    Prepare RGB batch for DINOv2.

    ``input_range``:
      - ``01``: already in [0, 1] (JiT after /255)
      - ``neg1_1``: SiT-style Normalize(0.5) images in [-1, 1]
      - ``uint8``: [0, 255]
    """
    if resolution is None:
        resolution = int(x.shape[-1])
    if input_range == "uint8":
        x = x.float() / 255.0
    elif input_range == "neg1_1":
        x = (x.float() + 1.0) * 0.5
    elif input_range == "01":
        x = x.float()
    else:
        raise ValueError(f"Unknown input_range={input_range!r}")

    mean = x.new_tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    target = 224 * (resolution // 256)
    if x.shape[-1] != target or x.shape[-2] != target:
        x = F.interpolate(x, size=(target, target), mode="bicubic", align_corners=False)
    return x


@torch.no_grad()
def extract_dinov2_patch_tokens(encoder, images: torch.Tensor) -> torch.Tensor:
    """Return ``x_norm_patchtokens`` with shape (B, T, D)."""
    feats = encoder.forward_features(images)
    if isinstance(feats, dict):
        if "x_norm_patchtokens" not in feats:
            raise KeyError(
                f"DINOv2 forward_features missing x_norm_patchtokens; keys={list(feats)}"
            )
        return feats["x_norm_patchtokens"]
    # Some wrappers return a tensor including CLS; drop CLS if present.
    if feats.ndim != 3:
        raise RuntimeError(f"Unexpected DINOv2 feature shape: {tuple(feats.shape)}")
    # Heuristic: if first token is CLS-sized sequence, drop it when T = 1 + H*W
    t = feats.shape[1]
    hw = int(round(t ** 0.5))
    if hw * hw == t - 1:
        return feats[:, 1:, :]
    return feats


def repa_neg_cosine_loss(
    z: torch.Tensor,
    z_tilde: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    REPA projection loss: mean over batch & tokens of ``-cos(z, z_tilde)``.

    ``z``: DINOv2 tokens (B, T, D), ``z_tilde``: projected SiT/JiT tokens (B, T, D).
    """
    if z.shape != z_tilde.shape:
        raise RuntimeError(
            f"REPA token shape mismatch: dino={tuple(z.shape)} vs proj={tuple(z_tilde.shape)}. "
            "Check image resolution / patch size / pos_embed resample."
        )
    z_n = F.normalize(z.float(), dim=-1, eps=eps)
    z_t = F.normalize(z_tilde.float(), dim=-1, eps=eps)
    return (-(z_n * z_t).sum(dim=-1)).mean()


def encoder_depth_to_layer_index(encoder_depth: int) -> int:
    """REPA uses 1-based ``encoder_depth``; convert to 0-based block index."""
    if encoder_depth < 1:
        raise ValueError(f"repa_encoder_depth must be >= 1, got {encoder_depth}")
    return int(encoder_depth) - 1
