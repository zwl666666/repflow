"""Training helpers for JiT distillation (checkpointing, logging, experiment dirs)."""

from __future__ import annotations

import argparse
import copy
import datetime
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

import util.misc as misc


def _saved_checkpoint_path(args: Any, epoch_name: Optional[str] = None) -> Path:
    if epoch_name is None or str(epoch_name) == "last":
        epoch_name = misc.LATEST_CHECKPOINT_STEM
    ckpt_root = Path(getattr(args, "checkpoint_dir", None) or args.output_dir)
    return ckpt_root / ("checkpoint-%s.pth" % epoch_name)


def _rclone_remote_checkpoint_path(args: Any, local_checkpoint_path: str) -> Optional[str]:
    remote_root = (getattr(args, "ckpt_rclone_remote", None) or "").strip().rstrip("/")
    if not remote_root:
        return None
    checkpoint_root = getattr(args, "checkpoint_root", None)
    if checkpoint_root:
        rel = os.path.relpath(local_checkpoint_path, checkpoint_root)
    else:
        rel = os.path.basename(local_checkpoint_path)
    return f"{remote_root}/{rel}".replace("\\", "/")


def sync_checkpoint_to_rclone(
    local_checkpoint_path: str,
    args: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    remote_path = _rclone_remote_checkpoint_path(args, local_checkpoint_path)
    if remote_path is None:
        return
    log = logger or logging.getLogger(__name__)
    try:
        proc = subprocess.run(
            ["rclone", "copyto", local_checkpoint_path, remote_path, "--retries", "3"],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if proc.returncode == 0:
            log.info("Synced checkpoint to rclone: %s", remote_path)
        else:
            stderr = (proc.stderr or proc.stdout or "").strip()
            log.warning("rclone sync failed (exit %d): %s", proc.returncode, stderr)
    except subprocess.TimeoutExpired:
        log.warning("rclone checkpoint sync timed out after 3600s")
    except FileNotFoundError:
        log.warning("rclone not found in PATH; skipped remote checkpoint sync")
    except Exception as exc:
        log.warning("rclone checkpoint sync error: %s", exc)


def save_model_and_sync(
    args: Any,
    model_without_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    logger: Optional[logging.Logger] = None,
    epoch_name: Optional[str] = None,
    student_ema: Optional[torch.nn.Module] = None,
) -> None:
    misc.save_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        epoch=epoch,
        epoch_name=epoch_name,
        student_ema=student_ema,
    )
    if not misc.is_main_process():
        if dist.is_initialized():
            dist.barrier()
        return

    checkpoint_path = _saved_checkpoint_path(args, epoch_name)
    sync_start = time.time()
    sync_checkpoint_to_rclone(str(checkpoint_path), args, logger)
    if logger and getattr(args, "ckpt_rclone_remote", None):
        logger.info("rclone sync finished in %.1fs", time.time() - sync_start)

    if dist.is_initialized():
        dist.barrier()


def get_effective_resume(args: Any) -> Optional[str]:
    """Return resume path from --resume or --ckpt (SiT-style alias)."""
    resume = getattr(args, "resume", None) or getattr(args, "ckpt", None)
    if resume is None or str(resume).strip() == "":
        return None
    return str(resume).strip()


def resolve_checkpoint_path(
    resume: Optional[str],
    resume_checkpoint: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a checkpoint file from:
      - explicit .pth file path
      - directory + optional checkpoint name (e.g. 'latest', 'last', '400', 'checkpoint-latest.pth')
      - directory: checkpoint-latest.pth, then legacy checkpoint-last / numbered files
    """
    if not resume:
        return None

    resume = os.path.expanduser(resume)
    if os.path.isfile(resume):
        return resume

    if not os.path.isdir(resume):
        return None

    if resume_checkpoint:
        name = resume_checkpoint
        if not name.endswith(".pth"):
            if name in ("latest", "last"):
                name = "checkpoint-latest.pth"
            elif name.isdigit():
                name = f"checkpoint-{name}.pth"
            elif not name.startswith("checkpoint-"):
                name = f"checkpoint-{name}.pth"
        path = os.path.join(resume, name)
        return path if os.path.isfile(path) else None

    for preferred in ("checkpoint-latest.pth", "checkpoint-last.pth", "latest.pth"):
        path = os.path.join(resume, preferred)
        if os.path.isfile(path):
            return path

    candidates = []
    for fname in os.listdir(resume):
        match = re.match(r"checkpoint-(\d+)\.pth$", fname)
        if match:
            candidates.append((int(match.group(1)), os.path.join(resume, fname)))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    return None


def create_logger(logging_dir: Optional[str], rank: Optional[int] = None):
    """Rank 0: log.txt + stdout; other ranks: log_rank{N}.txt for WARNING+."""
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    logger = logging.getLogger("jit_distill")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        handlers = [logging.StreamHandler()]
        if logging_dir:
            os.makedirs(logging_dir, exist_ok=True)
            handlers.append(logging.FileHandler(os.path.join(logging_dir, "log.txt")))
        for handler in handlers:
            handler.setLevel(logging.INFO)
            handler.setFormatter(fmt)
            logger.addHandler(handler)
        return logger

    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(logging_dir, f"log_rank{rank}.txt"))
        fh.setLevel(logging.WARNING)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def install_exception_logging(
    logger: logging.Logger,
    logging_dir: Optional[str],
    rank: int = 0,
) -> None:
    """Persist uncaught exceptions to log file(s) and stderr."""

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        header = f"[rank={rank}] Uncaught {exc_type.__name__}: {exc_value}"
        msg = f"{header}\n{tb_str}"

        try:
            logger.error(msg)
        except Exception:
            pass

        if logging_dir:
            os.makedirs(logging_dir, exist_ok=True)
            log_name = "log.txt" if rank == 0 else f"log_rank{rank}.txt"
            crash_path = os.path.join(logging_dir, log_name)
            try:
                with open(crash_path, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")
                    if not msg.endswith("\n"):
                        f.write("\n")
            except Exception:
                pass

        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def setup_training_paths(args: Any, rank: int) -> tuple[str, str]:
    """
    results_dir (--output_dir): logs, tensorboard, eval metrics only.
    checkpoint_dir: model checkpoints under /tmp (or --checkpoint_root).
    """
    if getattr(args, "results_dir", None):
        args.output_dir = os.path.abspath(args.results_dir)
    else:
        args.output_dir = os.path.abspath(args.output_dir)

    checkpoint_root = getattr(args, "checkpoint_root", "/tmp/jit_distill_checkpoints")
    experiment_name = os.path.basename(args.output_dir.rstrip("/"))
    checkpoint_dir = os.path.join(checkpoint_root, experiment_name)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    if misc.is_dist_avail_and_initialized():
        shared = [checkpoint_dir]
        dist.broadcast_object_list(shared, src=0)
        checkpoint_dir = shared[0]

    args.checkpoint_dir = checkpoint_dir
    return args.output_dir, checkpoint_dir


def setup_experiment_directory(args: Any, rank: int):
    """
    If --auto_experiment_dir is set, create results_dir/<index>-<model>-<student_model>/.
    Otherwise use args.output_dir as-is.
    """
    resume_path = resolve_checkpoint_path(
        get_effective_resume(args),
        getattr(args, "resume_checkpoint", None),
    )
    if resume_path is not None:
        checkpoint_root = os.path.abspath(
            getattr(args, "checkpoint_root", "/tmp/jit_distill_checkpoints")
        )
        parent = os.path.dirname(os.path.abspath(resume_path))
        if parent.startswith(checkpoint_root):
            experiment_dir = getattr(args, "results_dir", None) or args.output_dir
        else:
            experiment_dir = parent
        if rank == 0:
            os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir

    if getattr(args, "auto_experiment_dir", False):
        results_dir = getattr(args, "results_dir", None) or args.output_dir
        if rank == 0:
            os.makedirs(results_dir, exist_ok=True)
            experiment_index = len(glob(os.path.join(results_dir, "*")))
            model_name = str(args.model).replace("/", "-")
            if is_baseline_training(args):
                experiment_name = f"{experiment_index:03d}-{model_name}-baseline"
            else:
                student_name = str(getattr(args, "student_model", args.model)).replace("/", "-")
                experiment_name = f"{experiment_index:03d}-{model_name}-distill-{student_name}"
            experiment_dir = os.path.join(results_dir, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
        else:
            experiment_dir = None
        if misc.is_dist_avail_and_initialized():
            shared = [experiment_dir]
            dist.broadcast_object_list(shared, src=0)
            experiment_dir = shared[0]
        return experiment_dir

    experiment_dir = args.output_dir
    if rank == 0:
        os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def should_trigger_epoch_interval(args: Any, epoch: int) -> bool:
    every = getattr(args, "ckpt_every_epoch", None)
    if every is None:
        every = getattr(args, "save_freq", 0)
    return every > 0 and (epoch + 1) % every == 0


def compute_per_device_batch_size(args: Any) -> int:
    """Per-GPU batch size from --batch_size or --global_batch_size."""
    global_bs = getattr(args, "global_batch_size", None)
    if global_bs is not None:
        world_size = misc.get_world_size() if misc.is_dist_avail_and_initialized() else 1
        if global_bs % world_size != 0:
            raise ValueError(
                f"global_batch_size ({global_bs}) must be divisible by world_size ({world_size})."
            )
        return global_bs // world_size
    return args.batch_size


def load_checkpoint(
    checkpoint_path: str,
    model_without_ddp,
    optimizer,
    ema_device: torch.device,
    logger: Optional[logging.Logger] = None,
    student_ema=None,
):
    """Load model, EMA, and optionally optimizer; return start_epoch."""
    log = logger.info if logger is not None else print

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return {"start_epoch": 0, "loaded": False}

    # PyTorch 2.6+ defaults weights_only=True; our checkpoints include argparse.Namespace in 'args'.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_without_ddp.load_state_dict(checkpoint["model"])

    ema_state_dict = checkpoint.get("model_ema", checkpoint.get("model_ema1"))
    if ema_state_dict is None:
        raise KeyError("Checkpoint does not contain 'model_ema' or legacy 'model_ema1'")
    model_without_ddp.ema_params = [
        ema_state_dict[name].detach().to(ema_device).clone()
        for name, _ in model_without_ddp.named_parameters()
    ]

    if student_ema is not None:
        student_ema_state = checkpoint.get("student_ema")
        if student_ema_state is not None:
            student_ema.load_state_dict(student_ema_state)
            log("Loaded student_ema from checkpoint")
        else:
            sync_student_ema_from_main_ema(model_without_ddp, student_ema)
            log("Checkpoint missing student_ema; initialized from main EMA student weights")

    start_epoch = 0
    if "optimizer" in checkpoint and "epoch" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            log(f"Loaded optimizer state; resume from epoch {start_epoch}")
        except Exception as exc:
            log(f"Could not load optimizer state ({exc}); using fresh optimizer.")
    else:
        log("Checkpoint has no optimizer/epoch; model weights only.")

    log(f"Resumed checkpoint from {checkpoint_path}")
    log(f"EMA device: {ema_device}")
    del checkpoint
    return {"start_epoch": start_epoch, "loaded": True}


def init_ema_params(model_without_ddp, ema_device: torch.device):
    return [param.detach().to(ema_device).clone() for param in model_without_ddp.parameters()]


@torch.no_grad()
def update_ema_module(ema_model, model, decay: float = 0.9999) -> None:
    """Step a standalone EMA copy toward the current module (SiT train_utils.update_ema)."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


def build_student_ema(model_without_ddp, device: torch.device):
    """Separate frozen student copy for linear probe (not wrapped by denoiser DDP)."""
    if not hasattr(model_without_ddp, "net_student"):
        return None
    student_ema = copy.deepcopy(model_without_ddp.net_student)
    student_ema.to(device).eval()
    for param in student_ema.parameters():
        param.requires_grad_(False)
    return student_ema


def student_ema_decay(model_without_ddp) -> float:
    freq = max(1, int(getattr(model_without_ddp, "ema_update_freq", 1)))
    decay = float(getattr(model_without_ddp, "ema_decay", 0.9999))
    if freq <= 1:
        return decay
    return decay ** freq


@torch.no_grad()
def apply_ema_weights_from_checkpoint(
    denoiser,
    checkpoint: dict,
    *,
    skip_projector: bool = True,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Load EMA-smoothed ``net`` + ``net_student`` (and optional ``projector``) into denoiser.

    Uses ``checkpoint['model_ema']`` / ``model_ema1``. Student may fall back to
    ``checkpoint['student_ema']`` when ``model_ema`` has no ``net_student.*`` keys.
    """
    def _log(msg: str, *args):
        if logger is not None:
            logger.info(msg, *args)

    ema_state = checkpoint.get("model_ema") or checkpoint.get("model_ema1")
    if ema_state is None:
        raise RuntimeError("fd_use_ema=True but checkpoint has no model_ema or model_ema1")

    prefixes = ("net.", "net_student.")
    if not skip_projector:
        prefixes = (*prefixes, "projector.")

    partial = {
        key: value
        for key, value in ema_state.items()
        if any(key.startswith(prefix) for prefix in prefixes)
    }
    if not any(k.startswith("net.") for k in partial):
        raise RuntimeError("model_ema has no net.* weights")

    student_ema_state = checkpoint.get("student_ema")
    if student_ema_state is not None and hasattr(denoiser, "net_student"):
        prefix = "net_student."
        student_from_model_ema = any(k.startswith(prefix) for k in partial)
        if not student_from_model_ema:
            partial.update({f"{prefix}{k}": v for k, v in student_ema_state.items()})

    missing, unexpected = denoiser.load_state_dict(partial, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading EMA weights: {unexpected[:5]}")
    critical_missing = [
        k for k in missing
        if any(k.startswith(p) for p in ("net.", "net_student."))
    ]
    if critical_missing:
        raise RuntimeError(
            "model_ema missing required net/net_student keys: "
            + ", ".join(critical_missing[:5])
        )

    n_net = sum(1 for k in partial if k.startswith("net.") and not k.startswith("net_student."))
    n_student = sum(1 for k in partial if k.startswith("net_student."))
    _log(
        "[FD] Loaded EMA weights from model_ema: net=%d tensors, net_student=%d tensors",
        n_net,
        n_student,
    )
    return "model_ema"


@torch.no_grad()
def sync_student_ema_from_main_ema(model_without_ddp, student_ema) -> None:
    """Copy net_student EMA weights from denoiser.ema_params into student_ema."""
    if student_ema is None or model_without_ddp.ema_params is None:
        return
    target_state = student_ema.state_dict()
    ema_state = {}
    for i, (name, _param) in enumerate(model_without_ddp.named_parameters()):
        if not name.startswith("net_student."):
            continue
        key = name[len("net_student.") :]
        ema_state[key] = model_without_ddp.ema_params[i].to(
            device=target_state[key].device,
            dtype=target_state[key].dtype,
        )
    student_ema.load_state_dict(ema_state, strict=True)


@torch.no_grad()
def maybe_update_student_ema(student_ema, model_without_ddp, *, force: bool = False) -> bool:
    """Keep student_ema in sync when denoiser EMA is updated."""
    if student_ema is None or not hasattr(model_without_ddp, "net_student"):
        return False
    if not force and not getattr(model_without_ddp, "ema_params", None):
        return False
    decay = student_ema_decay(model_without_ddp)
    update_ema_module(student_ema, model_without_ddp.net_student, decay=decay)
    return True


def filter_state_dict_prefixes(state_dict: dict, exclude_prefixes: tuple[str, ...]) -> dict:
    if not exclude_prefixes:
        return state_dict
    return {
        key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in exclude_prefixes)
    }


def denoiser_load_exclude_prefixes(args: Any) -> tuple[str, ...]:
    if getattr(args, "skip_projector", False):
        return ("projector.",)
    return ()


def load_denoiser_state_dict(
    denoiser,
    state_dict: dict,
    *,
    exclude_prefixes: tuple[str, ...] = (),
    logger: Optional[logging.Logger] = None,
    strict: Optional[bool] = None,
):
    filtered = filter_state_dict_prefixes(state_dict, exclude_prefixes)
    use_strict = strict if strict is not None else not exclude_prefixes
    incompatible = denoiser.load_state_dict(filtered, strict=use_strict)
    if logger is not None and exclude_prefixes:
        logger.info(
            "Loaded denoiser checkpoint (skipped prefixes=%s, missing=%d, unexpected=%d)",
            exclude_prefixes,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
    return incompatible


def sit_main_dir() -> str:
    """SiT-main root (offline FID ref npz + guided-diffusion evaluator assets)."""
    return os.path.abspath(
        os.environ.get(
            "SIT_MAIN_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SiT-main"),
        )
    )


def apply_eval_defaults(args: Any) -> None:
    """Align eval paths with SiT-main/train.sh (env vars override, then SiT-main defaults)."""
    sit = sit_main_dir()
    if not getattr(args, "fid_ref_path", None):
        args.fid_ref_path = os.environ.get(
            "FID_REF_PATH",
            os.path.join(sit, "VIRTUAL_imagenet256_labeled.npz"),
        )
    if not getattr(args, "evaluator_script", None):
        args.evaluator_script = os.environ.get(
            "EVALUATOR_SCRIPT",
            os.path.join(sit, "evaluator.py"),
        )
    if not getattr(args, "eval_temp_root", None):
        args.eval_temp_root = os.environ.get("EVAL_TEMP_ROOT", "/tmp/jit_eval_tmp")
    args.fid_ref_path = os.path.abspath(os.path.expanduser(args.fid_ref_path))
    args.evaluator_script = os.path.abspath(os.path.expanduser(args.evaluator_script))


def evaluator_working_dir(evaluator_script: str) -> str:
    """CWD for guided-diffusion evaluator (Inception .pb is resolved relative to this)."""
    return os.path.dirname(os.path.abspath(evaluator_script))


def offline_eval_asset_paths(evaluator_script: str) -> dict[str, str]:
    sit = evaluator_working_dir(evaluator_script)
    return {
        "sit_main": sit,
        "evaluator_impl": os.path.join(sit, "_guided_diffusion_evaluator_impl.py"),
        "inception_v3": os.path.join(sit, "classify_image_graph_def.pb"),
    }


def check_offline_eval_assets(args: Any) -> list[str]:
    """Return human-readable errors if offline SiT eval assets are missing."""
    errors: list[str] = []
    if not os.path.isfile(args.fid_ref_path):
        errors.append(f"reference npz missing: {args.fid_ref_path}")
    if not os.path.isfile(args.evaluator_script):
        errors.append(f"evaluator bootstrap script missing: {args.evaluator_script}")
    assets = offline_eval_asset_paths(args.evaluator_script)
    if not os.path.isfile(assets["evaluator_impl"]):
        errors.append(
            "evaluator implementation missing: "
            f"{assets['evaluator_impl']} "
            f"(run: python {args.evaluator_script} --download-only)"
        )
    if not os.path.isfile(assets["inception_v3"]):
        errors.append(
            "InceptionV3 graph missing (offline): "
            f"{assets['inception_v3']} "
            "(copy from SiT-main or run evaluator once with network in SiT-main/)"
        )
    return errors


def log_offline_eval_config(logger: logging.Logger, args: Any) -> None:
    assets = offline_eval_asset_paths(args.evaluator_script)
    logger.info(
        "Offline FID eval (SiT-main): ref_npz=%s | evaluator=%s | cwd=%s | inception=%s",
        args.fid_ref_path,
        args.evaluator_script,
        assets["sit_main"],
        assets["inception_v3"],
    )
    missing = check_offline_eval_assets(args)
    if missing:
        for msg in missing:
            logger.warning("Offline eval asset: %s", msg)
    else:
        logger.info("Offline eval assets OK (no download required at eval time).")


def parse_comma_int_list(value: Optional[str], default=None):
    if value is None or str(value).strip() == "":
        return default
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def teacher_feature_layers_list(args: Any):
    layers = parse_comma_int_list(getattr(args, "teacher_feature_layers", None))
    if layers is not None:
        return layers
    return [5, 12, 19, 26]


def uniform_block_indices(depth: int, n: int) -> list:
    """Evenly spaced 0-based block indices; always includes the last block.

    For depth=24, n=4 this is ``[5, 11, 17, 23]`` (1/4, 2/4, 3/4, 4/4 of depth).
    """
    if n < 1:
        raise ValueError(f"Need at least 1 layer, got {n}")
    if depth < 1:
        raise ValueError(f"Student depth must be >= 1, got {depth}")
    n = min(int(n), int(depth))
    if n == 1:
        return [depth - 1]
    idxs = [int(round((i + 1) * depth / n - 1)) for i in range(n)]
    idxs = [min(max(i, 0), depth - 1) for i in idxs]
    out = []
    for i in idxs:
        if i not in out:
            out.append(i)
    if out[-1] != depth - 1:
        if depth - 1 in out:
            out.remove(depth - 1)
        out.append(depth - 1)
    return out


def projector_feature_layers_list(args: Any, student_depth: int):
    """Student block indices concatenated on the channel dim as projector input."""
    depth = int(student_depth)
    explicit = parse_comma_int_list(getattr(args, "projector_feature_layers", None))
    if explicit:
        resolved = []
        for fl in explicit:
            idx = fl if fl >= 0 else depth + fl
            if not (0 <= idx < depth):
                raise IndexError(
                    f"projector feature layer {fl} resolves to {idx}, "
                    f"out of range for student depth {depth}"
                )
            resolved.append(idx)
        last = depth - 1
        if last not in resolved:
            resolved.append(last)
        return resolved
    n = int(getattr(args, "projector_num_layers", 1) or 1)
    return uniform_block_indices(depth, n)


def _optional_override(args: Any, name: str):
    value = getattr(args, name, None)
    return value


def merge_init_kwargs(*parts: dict) -> dict:
    """Merge init kwargs left-to-right; later dicts override earlier ones."""
    merged = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged


def shared_vision_kwargs(args: Any) -> dict:
    """Common kwargs for teacher / student / projector registry factories."""
    return dict(
        input_size=args.img_size,
        in_channels=3,
        num_classes=args.class_num,
        attn_drop=args.attn_dropout,
        proj_drop=args.proj_dropout,
    )


def build_teacher_kwargs(args: Any) -> dict:
    """Architecture overrides only; shared args are passed in Denoiser.__init__."""
    # cosine=0 时不建 ProjLayer（teacher_cons_layer=None）
    cosine_w = float(getattr(args, "distill_cosine_weight", 0.1) or 0.0)
    repa_w = float(getattr(args, "repa_loss_weight", 0.0) or 0.0)
    kwargs = {
        "teacher_feature_layers": teacher_feature_layers_list(args),
        "teacher_cons_layer": (
            getattr(args, "teacher_cons_layer", 7) if cosine_w > 0 else None
        ),
        "teacher_repa_layer": (
            # REPA-main encoder_depth is 1-based; store 0-based block index on the model.
            int(getattr(args, "repa_encoder_depth", 8)) - 1 if repa_w > 0 else None
        ),
    }
    for key in (
        "depth",
        "hidden_size",
        "num_heads",
        "mlp_ratio",
        "bottleneck_dim",
        "in_context_len",
        "in_context_start",
        "patch_size",
    ):
        val = _optional_override(args, f"teacher_{key}")
        if val is not None:
            kwargs[key] = val
    if getattr(args, "uncondition", False):
        kwargs["in_context_len"] = 0
    return kwargs


def build_student_kwargs(args: Any) -> dict:
    overrides = {}
    for key in (
        "depth",
        "hidden_size",
        "num_heads",
        "mlp_ratio",
        "bottleneck_dim",
        "in_context_len",
        "in_context_start",
        "patch_size",
    ):
        val = _optional_override(args, f"student_{key}")
        if val is not None:
            overrides[key] = val
    if getattr(args, "student_mask_ratio", None) is not None:
        overrides["mask_ratio"] = args.student_mask_ratio
    return overrides


def build_projector_kwargs(args: Any) -> dict:
    """Collect kwargs for Projector_models[...](**kwargs)."""
    overrides = {}
    field_map = {
        "depth": "projector_dropout_depth",
        "hidden_size": "projector_hidden_size",
        "num_heads": "projector_num_heads",
        "mlp_ratio": "projector_mlp_ratio",
        "in_context_len": "projector_in_context_len",
        "in_context_start": "projector_in_context_start",
        "patch_size": "projector_patch_size",
        "block_dim": "projector_block_dim",
        "num_blocks": "projector_num_blocks",
        "out_dim": "projector_out_dim",
        "input_hidden_size": "projector_input_hidden_size",
        "t_embed_dim": "projector_t_embed_dim",
        "attn_drop": "projector_attn_dropout",
        "proj_drop": "projector_proj_dropout",
    }
    for dst, src in field_map.items():
        val = _optional_override(args, src)
        if val is not None:
            overrides[dst] = val
    # Backward-compatible alias (old flag name)
    legacy_blocks = _optional_override(args, "projector_depth")
    if legacy_blocks is not None and "num_blocks" not in overrides:
        overrides["num_blocks"] = legacy_blocks
    return overrides


def _count_params(module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _fmt_param_count(n: int) -> str:
    return f"{n:,} ({n / 1e6:.3f}M)"


def _log_module_children(logger, module, prefix: str = "", max_depth: int = 2, depth: int = 0) -> None:
    """Print submodule tree (limited depth to avoid huge logs)."""
    if depth >= max_depth:
        return
    for name, child in module.named_children():
        extra = ""
        if isinstance(child, torch.nn.ModuleList):
            extra = f" (len={len(child)})"
        logger.info("%s%s: %s%s", prefix, name, child.__class__.__name__, extra)
        _log_module_children(logger, child, prefix + "  ", max_depth, depth + 1)


def _summarize_teacher(logger, args: Any, teacher) -> None:
    logger.info("[Teacher] registry=%s", args.model)
    logger.info(
        "  image: size=%s patch=%s | transformer: depth=%s hidden=%s heads=%s",
        teacher.input_size,
        teacher.patch_size,
        teacher.depth,
        teacher.hidden_size,
        teacher.num_heads,
    )
    bottleneck = getattr(
        getattr(teacher.x_embedder, "proj1", None),
        "out_channels",
        getattr(teacher.x_embedder, "embed_dim", "?"),
    )
    logger.info(
        "  embed: num_patches=%s bottleneck=%s | in_context: len=%s start=%s%s",
        teacher.x_embedder.num_patches,
        bottleneck,
        teacher.in_context_len,
        teacher.in_context_start,
        " (disabled for --uncondition)" if getattr(args, "uncondition", False) else "",
    )
    if hasattr(teacher, "teacher_feature_layers"):
        logger.info(
            "  distill: feature_layers=%s | cons_layer=%s | repa_layer=%s",
            teacher.teacher_feature_layers,
            teacher.teacher_cons_layer,
            getattr(teacher, "teacher_repa_layer", None),
        )
    elif getattr(teacher, "teacher_repa_layer", None) is not None:
        logger.info("  REPA: teacher_repa_layer=%s", teacher.teacher_repa_layer)
    logger.info("  children:")
    _log_module_children(logger, teacher, prefix="    ", max_depth=2)


def _summarize_student(logger, args: Any, student) -> None:
    depth = len(student.blocks)
    logger.info("[Student] registry=%s", args.student_model)
    logger.info(
        "  image: size=%s patch=%s | transformer: depth=%s hidden=%s heads=%s",
        student.input_size,
        student.patch_size,
        depth,
        student.hidden_size,
        student.num_heads,
    )
    num_patches = getattr(
        student.x_embedder,
        "num_patches",
        getattr(student.x_embedder.x_embedder, "num_patches", "?"),
    )
    logger.info(
        "  embed: num_patches=%s mask_ratio=%s",
        num_patches,
        student.x_embedder.mask_ratio,
    )
    logger.info("  children:")
    _log_module_children(logger, student, prefix="    ", max_depth=2)


def _summarize_projector(logger, args: Any, projector) -> None:
    attn_drop = projector.blocks[0].attn.attn_drop.p if len(projector.blocks) else 0.0
    proj_drop = projector.blocks[0].mlp.ffn_dropout.p if len(projector.blocks) else 0.0
    logger.info("[Projector] registry=%s", getattr(args, "projector_model", None) or args.student_model)
    logger.info(
        "  image: size=%s patch=%s num_patches=%s",
        projector.input_size,
        projector.patch_size,
        projector.num_patches,
    )
    logger.info(
        "  transformer: num_blocks=%s block_dim=%s heads=%s mlp_ratio=%s",
        projector.num_blocks,
        projector.block_dim,
        projector.num_heads,
        projector.mlp_ratio,
    )
    logger.info(
        "  io: in_bottle %s->%s | out_bottle %s->%s",
        projector.in_bottle.in_features,
        projector.in_bottle.out_features,
        projector.out_bottle.in_features,
        projector.out_bottle.out_features,
    )
    logger.info(
        "  t_embed: dim=%s | dropout: attn=%.3f proj=%.3f",
        projector.t_embedder.mlp[0].in_features,
        attn_drop,
        proj_drop,
    )
    logger.info("  children:")
    _log_module_children(logger, projector, prefix="    ", max_depth=2)


def is_baseline_training(args: Any) -> bool:
    return getattr(args, "training_mode", "distill") == "baseline"


def build_denoiser(args: Any):
    """Factory: teacher-only baseline vs full distillation denoiser."""
    if is_baseline_training(args):
        from denoiser_baseline import Denoiser
    else:
        from denoiser_distill import Denoiser
    return Denoiser(args)


def log_model_architecture(logger, args: Any, denoiser) -> None:
    """Print model structure: hyper-parameters + submodule tree."""
    if logger is None:
        return
    mode = "baseline (flow only)" if is_baseline_training(args) else "distill"
    logger.info("=== Model architecture [%s] ===", mode)
    _summarize_teacher(logger, args, denoiser.net)
    if hasattr(denoiser, "net_student"):
        _summarize_student(logger, args, denoiser.net_student)
        if getattr(denoiser, "projector", None) is not None:
            _summarize_projector(logger, args, denoiser.projector)
            layers = getattr(denoiser, "projector_feature_layers", None)
            if layers:
                logger.info(
                    "  projector student layers (concat dim=-1): %s (input dim=%s)",
                    layers,
                    denoiser.net_student.hidden_size * len(layers),
                )
        else:
            logger.info("[Projector] skipped (not loaded)")
    logger.info(
        "[Denoiser] flow: P_mean=%s P_std=%s t_eps=%s | sampling: %s steps=%s cfg=%s",
        denoiser.P_mean,
        denoiser.P_std,
        denoiser.t_eps,
        denoiser.method,
        denoiser.steps,
        denoiser.cfg_scale,
    )
    logger.info("  loss weights: flow=%.4f", denoiser.flow_loss_weight)
    if hasattr(denoiser, "distill_loss_weight"):
        logger.info(
            "  distill: weight=%.4f mse=%.4f cosine=%.4f",
            denoiser.distill_loss_weight,
            denoiser.distill_mse_weight,
            denoiser.distill_cosine_weight,
        )
    repa_w = float(getattr(denoiser, "repa_loss_weight", 0.0) or 0.0)
    if repa_w > 0:
        logger.info(
            "  REPA: weight=%.4f enc=%s projector=%s",
            repa_w,
            getattr(denoiser, "repa_enc_type", None),
            "yes" if getattr(denoiser, "repa_projector", None) is not None else "no",
        )
    sra_w = float(getattr(denoiser, "sra_loss_weight", 0.0) or 0.0)
    if sra_w > 0:
        logger.info(
            "  SRA: weight=%.4f s=%s t=%s t_max=%s loss=%s ap_head=%s",
            sra_w,
            getattr(denoiser, "sra_block_s", None),
            getattr(denoiser, "sra_block_t", None),
            getattr(denoiser, "sra_t_max", None),
            getattr(denoiser, "sra_loss_type", None),
            "yes" if getattr(denoiser, "ap_head", None) is not None else "no",
        )
    logger.info("==========================")


def log_model_parameters(logger, denoiser) -> None:
    """Print parameter counts for teacher / student / projector (SiT-style)."""
    if logger is None:
        return

    teacher = denoiser.net
    teacher_total = _count_params(teacher)
    teacher_train = _count_params(teacher, trainable_only=True)
    total = _count_params(denoiser)
    total_train = _count_params(denoiser, trainable_only=True)

    logger.info("=== Model parameters ===")
    logger.info(
        "Teacher (net):     total=%s | trainable=%s",
        _fmt_param_count(teacher_total),
        _fmt_param_count(teacher_train),
    )
    if hasattr(denoiser, "net_student"):
        student = denoiser.net_student
        student_total = _count_params(student)
        student_train = _count_params(student, trainable_only=True)
        logger.info(
            "Student:           total=%s | trainable=%s",
            _fmt_param_count(student_total),
            _fmt_param_count(student_train),
        )
        if getattr(denoiser, "projector", None) is not None:
            projector = denoiser.projector
            projector_total = _count_params(projector)
            projector_train = _count_params(projector, trainable_only=True)
            logger.info(
                "Projector:         total=%s | trainable=%s",
                _fmt_param_count(projector_total),
                _fmt_param_count(projector_train),
            )
        else:
            logger.info("Projector:         skipped (not loaded)")
    if getattr(denoiser, "repa_projector", None) is not None:
        rp = denoiser.repa_projector
        logger.info(
            "REPA projector:    total=%s | trainable=%s",
            _fmt_param_count(_count_params(rp)),
            _fmt_param_count(_count_params(rp, trainable_only=True)),
        )
    if getattr(denoiser, "ap_head", None) is not None:
        head = denoiser.ap_head
        logger.info(
            "SRA ap_head:       total=%s | trainable=%s",
            _fmt_param_count(_count_params(head)),
            _fmt_param_count(_count_params(head, trainable_only=True)),
        )
    logger.info(
        "Denoiser (all):    total=%s | trainable=%s",
        _fmt_param_count(total),
        _fmt_param_count(total_train),
    )
    logger.info("========================")


def log_model_summary(logger, args: Any, denoiser) -> None:
    """Architecture + parameter counts."""
    log_model_architecture(logger, args, denoiser)
    log_model_parameters(logger, denoiser)


def log_projector_config(logger, args: Any, projector) -> None:
    """Deprecated: use log_model_architecture."""
    _summarize_projector(logger, args, projector)


def add_model_structure_args(parser: argparse.ArgumentParser):
    """SiT-style overrides for teacher / student / projector architecture."""
    teacher = parser.add_argument_group("Teacher architecture overrides")
    teacher.add_argument(
        "--teacher_feature_layers",
        type=str,
        default=None,
        help="Comma-separated block indices for teacher DIFT features, e.g. '5,12,19,26'.",
    )
    teacher.add_argument(
        "--teacher_cons_layer",
        type=int,
        default=7,
        help="Teacher block index for fea_cons (cosine alignment branch).",
    )
    for name, help_suffix in [
        ("depth", "transformer depth"),
        ("hidden_size", "hidden dimension"),
        ("num_heads", "attention heads"),
        ("mlp_ratio", "MLP ratio"),
        ("bottleneck_dim", "patch embed bottleneck dim"),
        ("in_context_len", "in-context token length"),
        ("in_context_start", "block index to insert in-context tokens"),
        ("patch_size", "patch size"),
    ]:
        teacher.add_argument(
            f"--teacher_{name}",
            type=int if name != "mlp_ratio" else float,
            default=None,
            help=f"Override teacher {help_suffix} (default: from --model registry).",
        )

    student = parser.add_argument_group("Student architecture")
    student.add_argument(
        "--student_mask_ratio",
        type=float,
        default=None,
        help="Random patch mask ratio for student (default: 0.75 in model).",
    )
    for name, typ, help_suffix in [
        ("depth", int, "transformer depth"),
        ("hidden_size", int, "hidden dimension"),
        ("num_heads", int, "attention heads"),
        ("mlp_ratio", float, "MLP ratio"),
        ("patch_size", int, "patch size"),
        ("bottleneck_dim", int, "patch embed bottleneck dim"),
    ]:
        student.add_argument(
            f"--student_{name}",
            type=typ,
            default=None,
            help=f"Override student {help_suffix} (default: from --student_model registry).",
        )

    projector = parser.add_argument_group("Projector architecture (SiT-style)")
    projector.add_argument(
        "--projector_model",
        type=str,
        default=None,
        help="Registry key in Projector_models (default: same as --student_model).",
    )
    projector.add_argument(
        "--projector_hidden_size",
        type=int,
        default=None,
        help="Projector registry hidden_size / label dim (default: from --projector_model).",
    )
    projector.add_argument(
        "--projector_input_hidden_size",
        type=int,
        default=None,
        help="Student token dim fed to in_bottle (default: student hidden_size).",
    )
    projector.add_argument(
        "--projector_block_dim",
        type=int,
        default=None,
        help="Internal transformer width (default: 384).",
    )
    projector.add_argument(
        "--projector_num_blocks",
        type=int,
        default=None,
        help="Number of projector JiT blocks (default: 8).",
    )
    projector.add_argument(
        "--projector_dropout_depth",
        type=int,
        default=None,
        help="Reference depth for middle-block dropout schedule (default: registry depth, e.g. 12).",
    )
    projector.add_argument(
        "--projector_depth",
        type=int,
        default=None,
        help="Deprecated alias for --projector_num_blocks.",
    )
    projector.add_argument(
        "--projector_num_heads",
        type=int,
        default=None,
        help="Attention heads inside projector blocks (default: from registry).",
    )
    projector.add_argument(
        "--projector_mlp_ratio",
        type=float,
        default=None,
        help="MLP ratio in projector blocks (default: 4.0).",
    )
    projector.add_argument(
        "--projector_out_dim",
        type=int,
        default=None,
        help="Output dim after out_bottle (default: 4608 = 1152*4).",
    )
    projector.add_argument(
        "--projector_t_embed_dim",
        type=int,
        default=None,
        help="Timestep MLP width (default: same as --projector_block_dim).",
    )
    projector.add_argument(
        "--projector_attn_dropout",
        type=float,
        default=None,
        help="Attention dropout in projector blocks (default: --attn_dropout).",
    )
    projector.add_argument(
        "--projector_proj_dropout",
        type=float,
        default=None,
        help="Projection/FFN dropout in projector blocks (default: --proj_dropout).",
    )
    projector.add_argument(
        "--projector_in_context_len",
        type=int,
        default=None,
        help="Unused in forward; kept for registry compatibility.",
    )
    projector.add_argument(
        "--projector_in_context_start",
        type=int,
        default=None,
        help="Unused in forward; kept for registry compatibility.",
    )
    projector.add_argument(
        "--projector_patch_size",
        type=int,
        default=None,
        help="Patch size for projector pos embed (default: from registry / img_size).",
    )
    projector.add_argument(
        "--projector_num_layers",
        type=int,
        default=1,
        help="How many student blocks to concat on the channel dim as projector input. "
        "Layers are chosen uniformly and always include the last block "
        "(depth=24, n=4 → 5,11,17,23). Default 1 is last layer only.",
    )
    projector.add_argument(
        "--projector_feature_layers",
        type=str,
        default=None,
        help="Comma-separated student block indices concatenated on the last dim as "
        "projector input, e.g. '5,11,17,23'. Negative indices allowed. The last "
        "student block is always included. When set, overrides --projector_num_layers.",
    )


def add_training_mode_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Training mode")
    group.add_argument(
        "--training_mode",
        type=str,
        default="distill",
        choices=("distill", "baseline"),
        help="distill: teacher+student+projector; baseline: teacher flow loss only.",
    )


def add_loss_weight_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Loss weights")
    group.add_argument(
        "--flow_loss_weight",
        type=float,
        default=1.0,
        help="Weight on flow-matching velocity MSE loss.",
    )
    group.add_argument(
        "--distill_loss_weight",
        type=float,
        default=1.0,
        help="Global weight on distillation loss added to total loss.",
    )
    group.add_argument(
        "--distill_mse_weight",
        type=float,
        default=1.0,
        help="Weight on masked smooth-L1 between student projector output and teacher features.",
    )
    group.add_argument(
        "--distill_cosine_weight",
        type=float,
        default=0.1,
        help="Weight on cosine loss between clean student anchor and teacher fea_cons.",
    )
    group.add_argument(
        "--student_loss_weight",
        type=float,
        default=None,
        help="Alias for --distill_loss_weight (SiT naming).",
    )


def add_repa_args(parser: argparse.ArgumentParser):
    """Optional REPA (DINOv2) alignment. Default weight=0 keeps old configs unchanged."""
    group = parser.add_argument_group("REPA (DINOv2 alignment)")
    group.add_argument(
        "--repa_loss_weight",
        type=float,
        default=0.0,
        help="Weight on REPA neg-cosine loss. Default 0 disables DINOv2 load / projector / hook.",
    )
    group.add_argument(
        "--repa_enc_type",
        type=str,
        default="dinov2-vit-b",
        help="Vision encoder type, e.g. dinov2-vit-b / dinov2-vit-l / dinov2-vit-g.",
    )
    group.add_argument(
        "--repa_encoder_depth",
        type=int,
        default=8,
        help="1-based teacher block depth for REPA (REPA-main default 8 → 0-based index 7).",
    )
    group.add_argument(
        "--repa_projector_dim",
        type=int,
        default=2048,
        help="Hidden width of the 3-layer REPA MLP projector.",
    )
    group.add_argument(
        "--repa_encoder_repo",
        type=str,
        default=None,
        help="Optional local path to facebookresearch/dinov2 repo for torch.hub local load.",
    )
    group.add_argument(
        "--repa_encoder_ckpt",
        type=str,
        default=None,
        help="Optional local DINOv2 checkpoint (.pth). If unset, hub pretrained weights are used.",
    )


def sra_layer_index(block_out_1based: int) -> int:
    """SRA --block-out-* is 1-based (after block i when i+1 == ad). Convert to 0-based."""
    idx = int(block_out_1based) - 1
    if idx < 0:
        raise ValueError(f"SRA block-out must be >= 1, got {block_out_1based}")
    return idx


def sra_alignment_loss(online_feat, teacher_feat, loss_type="cos"):
    """SRA-main Simpleloss on token features (N, T, D). Cosine uses last dim."""
    if loss_type == "cos":
        return (1.0 - F.cosine_similarity(online_feat, teacher_feat, dim=-1)).mean()
    if loss_type == "sml1":
        return F.smooth_l1_loss(online_feat, teacher_feat, beta=0.05)
    if loss_type == "l2":
        return F.mse_loss(online_feat, teacher_feat)
    if loss_type == "l1":
        return F.l1_loss(online_feat, teacher_feat)
    raise NotImplementedError(f"Unknown SRA loss type: {loss_type}")


def add_sra_args(parser: argparse.ArgumentParser):
    """Optional SRA (self-representation alignment). Default weight=0 keeps existing scripts unchanged."""
    group = parser.add_argument_group("SRA (self-representation alignment)")
    group.add_argument(
        "--sra_loss_weight",
        type=float,
        default=0.0,
        help="Weight on SRA alignment. Default 0 disables SRA. SiT-SRA cosine uses 0.5. "
        "Requires --training_mode baseline (no student).",
    )
    group.add_argument(
        "--sra_block_out_s",
        type=int,
        default=8,
        help="1-based earlier block for the online model (paper XL: 8 → 0-based 7).",
    )
    group.add_argument(
        "--sra_block_out_t",
        type=int,
        default=20,
        help="1-based later block for the EMA teacher (paper XL: 20 → 0-based 19).",
    )
    group.add_argument(
        "--sra_t_max",
        type=float,
        default=0.2,
        help="Max time gap. JiT uses t=0 noise / t=1 data, so EMA is at t + t_max * U(0,1).",
    )
    group.add_argument(
        "--sra_loss_type",
        type=str,
        default="cos",
        choices=("cos", "sml1", "l2", "l1"),
        help="SRA alignment loss. Paper used sml1; README prefers cosine.",
    )


def apply_loss_weight_aliases(args: Any):
    if getattr(args, "student_loss_weight", None) is not None:
        args.distill_loss_weight = args.student_loss_weight


def add_distill_training_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Training / checkpointing")
    group.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size across GPUs; overrides --batch_size when set.",
    )
    group.add_argument(
        "--global_seed",
        type=int,
        default=None,
        help="Base seed (per-rank offset applied in main); overrides --seed when set.",
    )
    group.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Persistent results root (logs, eval/); checkpoints go to --checkpoint_root.",
    )
    group.add_argument(
        "--checkpoint_root",
        type=str,
        default="/tmp/jit_distill_checkpoints",
        help="Root directory for model checkpoints (default: /tmp/jit_distill_checkpoints).",
    )
    group.add_argument(
        "--ckpt_rclone_remote",
        type=str,
        default=None,
        help="Optional rclone remote root (e.g. tf:gugroup/zwl). After each checkpoint save, "
        "upload via rclone copyto to <remote>/<experiment>/<file>, mirroring --checkpoint_root.",
    )
    group.add_argument(
        "--auto_experiment_dir",
        action="store_true",
        help="Create output_dir as results_dir/<idx>-<model>-distill-<student_model>/.",
    )
    group.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint .pth file, or directory containing checkpoint-*.pth.",
    )
    group.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Alias for --resume (explicit checkpoint file path).",
    )
    group.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="When --resume is a directory: checkpoint name (e.g. latest, last, 400, checkpoint-latest.pth).",
    )
    group.add_argument(
        "--ckpt_every_epoch",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (default: use --save_freq).",
    )
    group.add_argument(
        "--save_last_every_epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save checkpoint-last.pth every epoch (in addition to periodic saves).",
    )
