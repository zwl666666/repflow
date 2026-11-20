"""FDr-6 evaluation for JiT periodic eval (FD-Loss repr models, offline weights)."""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import sys
from typing import Any

import torch

from cuda_memory_utils import release_eval_cuda_memory

logger = logging.getLogger("jit_fdr6")

DEFAULT_FD_LOSS_ROOT = "./FD-Loss"
DEFAULT_REPR_WEIGHTS_ROOT = "/tmp/hf_timm"

FDR6_MODEL_SPECS: list[tuple[str, str]] = [
    ("inception", "FID(ADM)"),
    ("convnext", "convnext"),
    ("vit_large_patch16_224.mae", "mae_cls"),
    ("vit_large_patch14_dinov2.lvd142m", "dinov2_cls"),
    ("vit_so400m_patch16_siglip_256.v2_webli", "siglip_cls"),
    ("vit_large_patch14_clip_224.openai", "clip_cls"),
]

FDR6_VALIDATION_FD = {
    "FID(ADM)": 1.68,
    "convnext": 56.87,
    "dinov2_cls": 14.19,
    "mae_cls": 0.04,
    "siglip_cls": 0.60,
    "clip_cls": 5.60,
}

REQUIRED_REF_STATS = [
    "guided_diffusion_stats.npz",
    "convnext_in256_t224_stats.npz",
    "vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz",
    "vit_large_patch16_224_mae_in256_t224_stats.npz",
    "vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz",
    "vit_large_patch14_clip_224_openai_in256_t256_stats.npz",
]

WEIGHT_SUBDIRS = ("convnext", "mae", "dinov2", "siglip", "clip")


def _repr_weights_root(args) -> str:
    return getattr(args, "fdr6_repr_weights_root", None) or os.environ.get(
        "FD_REPR_WEIGHTS_ROOT", DEFAULT_REPR_WEIGHTS_ROOT
    )


def _inception_weights_path(args) -> str:
    explicit = getattr(args, "fdr6_inception_weights", None)
    if explicit:
        return explicit
    env_path = os.environ.get("FD_INCEPTION_WEIGHTS", "").strip()
    if env_path:
        return env_path
    return os.path.join(_repr_weights_root(args), "weights-inception-2015-12-05-6726825d.pth")


def check_fdr6_assets(args) -> list[str]:
    """Return missing paths required for FDr-6 (empty list => ready)."""
    missing: list[str] = []
    fd_root = getattr(args, "fdr6_fd_loss_root", DEFAULT_FD_LOSS_ROOT)
    if not os.path.isdir(fd_root):
        missing.append(fd_root)
        return missing

    stats_dir = os.path.join(fd_root, "data", "fid_stats")
    for name in REQUIRED_REF_STATS:
        path = os.path.join(stats_dir, name)
        if not os.path.isfile(path):
            missing.append(path)

    weights_root = _repr_weights_root(args)
    inception = _inception_weights_path(args)
    if not os.path.isfile(inception):
        missing.append(inception)
    for subdir in WEIGHT_SUBDIRS:
        path = os.path.join(weights_root, subdir)
        if not os.path.isdir(path):
            missing.append(path)
    return missing


def _module_root(module) -> str | None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    return os.path.abspath(os.path.dirname(module_file))


def _purge_modules_under(root: str, prefixes: tuple[str, ...]) -> dict[str, Any]:
    """Remove cached modules whose __file__ lives under `root`; return saved mapping."""
    root = os.path.abspath(root)
    saved: dict[str, Any] = {}
    for key in list(sys.modules.keys()):
        if not any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes):
            continue
        mod = sys.modules.get(key)
        mod_root = _module_root(mod)
        if mod_root is None or not mod_root.startswith(root):
            continue
        saved[key] = mod
        del sys.modules[key]
    return saved


@contextlib.contextmanager
def _fd_loss_import_context(fd_loss_root: str):
    """Run FD-Loss eval code without colliding with JiT's `frechet_distance` package."""
    fd_loss_root = os.path.abspath(fd_loss_root)
    jit_root = os.path.abspath(os.path.dirname(__file__))
    saved_path = sys.path.copy()
    saved_jit_modules = _purge_modules_under(
        jit_root,
        ("frechet_distance", "eval_all_fds"),
    )

    # JiT ships a regular `frechet_distance` package that wins over FD-Loss even when
    # FD-Loss is first on sys.path, so drop JiT from the import path for this block.
    sys.path = [p for p in sys.path if os.path.abspath(p) != jit_root]
    if fd_loss_root not in sys.path:
        sys.path.insert(0, fd_loss_root)

    prev_cwd = os.getcwd()
    os.chdir(fd_loss_root)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        _purge_modules_under(
            fd_loss_root,
            ("frechet_distance", "eval_all_fds", "utils"),
        )
        sys.path[:] = saved_path
        sys.modules.update(saved_jit_modules)


def _setup_offline_env(args) -> None:
    weights_root = _repr_weights_root(args)
    os.environ["FD_REPR_WEIGHTS_ROOT"] = weights_root
    os.environ["FD_INCEPTION_WEIGHTS"] = _inception_weights_path(args)


def _setup_offline_env(args) -> None:
    weights_root = _repr_weights_root(args)
    os.environ["FD_REPR_WEIGHTS_ROOT"] = weights_root
    os.environ["FD_INCEPTION_WEIGHTS"] = _inception_weights_path(args)


def _unload_repr_models(repr_models) -> None:
    for entry in repr_models or []:
        model = entry.get("model")
        if isinstance(model, torch.nn.Module):
            model.cpu()
        for key in ("mu_ref", "sigma_ref"):
            tensor = entry.get(key)
            if isinstance(tensor, torch.Tensor):
                entry[key] = tensor.cpu()


@contextlib.contextmanager
def _single_process_fdr6_eval():
    """Run FD-Loss folder eval on rank 0 only (ignore torch.distributed).

    Must patch names bound in ``eval_all_fds`` (``from utils... import ...``),
    not just ``utils.distributed_util`` module attributes.
    """
    import eval_all_fds as eaf

    stubs = {
        "get_world_size": lambda: 1,
        "get_global_rank": lambda: 0,
        "is_enabled": lambda: False,
        "broadcast_scalar": lambda value, device="cuda": value,
    }
    saved = {name: getattr(eaf, name) for name in stubs}
    for name, fn in stubs.items():
        setattr(eaf, name, fn)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(eaf, name, fn)


def _evaluate_fdr6_models(
    sample_dir: str,
    *,
    img_size: int,
    batch_size: int,
    num_workers: int,
    log: logging.Logger,
) -> dict[str, float]:
    from eval_all_fds import (  # noqa: WPS433
        DEFAULT_TARGET_SIZES,
        _evaluate_from_folder,
        _load_repr_models,
    )

    fd_results: dict[str, float] = {}
    for model_name, result_name in FDR6_MODEL_SPECS:
        log.info("[FDr-6] repr model=%s metric=%s", model_name, result_name)
        repr_models = _load_repr_models(
            [model_name],
            img_size,
            target_size_overrides=DEFAULT_TARGET_SIZES,
        )
        repr_models = [entry for entry in repr_models if entry["name"] == result_name]
        if not repr_models:
            raise RuntimeError(f"No repr entry for {model_name} -> {result_name}")

        with _single_process_fdr6_eval():
            log.info("[FDr-6] %s: extracting features + computing FD ...", result_name)
            results, _, elapsed, num_images, _, _ = _evaluate_from_folder(
                repr_models,
                sample_dir,
                img_size=img_size,
                batch_size=batch_size,
                num_workers=num_workers,
                prc_ref_features=None,
                prc_model_names=[],
            )
        if result_name not in results:
            raise RuntimeError(
                f"Missing FD for {result_name}; got keys={list(results.keys())}"
            )
        fd_val = float(results[result_name])
        fd_results[result_name] = fd_val
        log.info(
            "[FDr-6] %s: fd=%.4f fdr=%.4f (%.1fs, n=%d)",
            result_name,
            fd_val,
            fd_val / FDR6_VALIDATION_FD[result_name],
            elapsed,
            num_images,
        )
        _unload_repr_models(repr_models)
        del repr_models, results
        release_eval_cuda_memory(log, reset_compile=True)
    return fd_results


def evaluate_fdr6_from_folder(
    sample_dir: str,
    args,
    logger_obj: logging.Logger | None = None,
) -> dict[str, Any]:
    """Compute FDr-6 from eval PNG folder (call on rank 0 only)."""
    log = logger_obj or logger
    missing = check_fdr6_assets(args)
    if missing:
        raise FileNotFoundError("FDr-6 assets missing:\n  - " + "\n  - ".join(missing))

    _setup_offline_env(args)
    fd_root = getattr(args, "fdr6_fd_loss_root", DEFAULT_FD_LOSS_ROOT)
    img_size = getattr(args, "img_size", 256)
    batch_size = getattr(args, "fdr6_batch_size", 64)
    num_workers = getattr(args, "fdr6_num_workers", 8)

    with _fd_loss_import_context(fd_root):
        fd_results = _evaluate_fdr6_models(
            sample_dir,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            log=log,
        )
    release_eval_cuda_memory(log, reset_compile=True)

    per_model = {
        result_name: {
            "fd": fd_results[result_name],
            "fdr": fd_results[result_name] / FDR6_VALIDATION_FD[result_name],
            "val_fd": FDR6_VALIDATION_FD[result_name],
        }
        for _, result_name in FDR6_MODEL_SPECS
    }
    fdr6 = float(sum(entry["fdr"] for entry in per_model.values()) / len(per_model))
    log.info("[FDr-6] aggregate fdr6=%.4f", fdr6)
    return {
        "fdr6": fdr6,
        "fdr6_per_model": per_model,
    }
