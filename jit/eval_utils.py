"""Periodic FID/IS evaluation (guided-diffusion evaluator), aligned with SiT-main/train.py."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

import util.misc as misc
from cuda_memory_utils import release_eval_cuda_memory
from fdr6_eval import check_fdr6_assets, evaluate_fdr6_from_folder
from train_utils import (
    check_offline_eval_assets,
    evaluator_working_dir,
    should_trigger_epoch_interval,
    sync_student_ema_from_main_ema,
)


def should_trigger_linear_eval(args: Any, epoch: int) -> bool:
    start_epoch = getattr(args, "linear_eval_start_epoch", 1)
    if epoch + 1 < start_epoch:
        return False
    return should_trigger_epoch_interval(args, epoch)


def add_eval_args(parser):
    group = parser.add_argument_group("Evaluation (SiT-style)")
    group.add_argument(
        "--dist_timeout_minutes",
        type=float,
        default=180.0,
        help=(
            "NCCL collective timeout in minutes for init_process_group. Must exceed "
            "the rank-0-only FID evaluator runtime, otherwise idle ranks trip the "
            "watchdog and abort the run at the eval stage."
        ),
    )
    group.add_argument(
        "--eval_during_train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run guided-diffusion evaluator every --ckpt_every_epoch.",
    )
    group.add_argument(
        "--eval_num_fid_samples",
        type=int,
        default=50000,
        help="Number of images for FID eval (same as SiT).",
    )
    group.add_argument(
        "--eval_per_proc_batch_size",
        type=int,
        default=None,
        help="Generation batch size per GPU for eval (default: --gen_bsz).",
    )
    group.add_argument(
        "--eval_temp_root",
        type=str,
        default="/tmp/jit_eval_tmp",
        help="Root for temporary eval PNG/NPZ (cleaned after metrics).",
    )
    group.add_argument(
        "--fid_ref_path",
        type=str,
        default=None,
        help="Reference batch npz for guided-diffusion evaluator.py.",
    )
    group.add_argument(
        "--evaluator_script",
        type=str,
        default=None,
        help="Path to guided-diffusion evaluations/evaluator.py.",
    )
    group.add_argument(
        "--eval_log_interval",
        type=int,
        default=20,
        help="Log sampling progress every N iterations.",
    )
    group.add_argument(
        "--keep_eval_samples",
        action="store_true",
        help="Keep temporary PNG/NPZ under --eval_temp_root after eval.",
    )
    group.add_argument(
        "--eval_fdr6",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run FDr-6 (FD-Loss 6-repr metric) on periodic eval samples.",
    )
    group.add_argument(
        "--fdr6_fd_loss_root",
        type=str,
        default="./FD-Loss",
        help="FD-Loss repo root (for ref stats and eval code).",
    )
    group.add_argument(
        "--fdr6_repr_weights_root",
        type=str,
        default="/tmp/hf_timm",
        help="Local timm/Inception weights root (rclone hf_timm).",
    )
    group.add_argument(
        "--fdr6_inception_weights",
        type=str,
        default=None,
        help="Override Inception .pth path (default: <fdr6_repr_weights_root>/weights-inception-*.pth).",
    )
    group.add_argument("--fdr6_batch_size", type=int, default=64)
    group.add_argument("--fdr6_num_workers", type=int, default=8)
    group.add_argument(
        "--linear_eval_during_train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run linear probe on frozen student every --ckpt_every_epoch "
        "(only when --uncondition and distillation is enabled).",
    )
    group.add_argument(
        "--linear_train_data_path",
        type=str,
        default=None,
        help="Image root for linear probe training set.",
    )
    group.add_argument(
        "--linear_val_data_path",
        type=str,
        default=None,
        help="Image root for linear probe validation set.",
    )
    group.add_argument(
        "--linear_train_label_file",
        type=str,
        default=None,
        help="Label file for linear probe training set.",
    )
    group.add_argument(
        "--linear_val_label_file",
        type=str,
        default=None,
        help="Label file for linear probe validation set.",
    )
    group.add_argument("--linear_epochs", type=int, default=30)
    group.add_argument(
        "--linear_eval_start_epoch",
        type=int,
        default=1,
        help="First training epoch (1-based) at which linear probe eval may run.",
    )
    group.add_argument("--linear_global_batch_size", type=int, default=1024)
    group.add_argument("--linear_lr", type=float, default=1e-3)
    group.add_argument("--linear_num_workers", type=int, default=8)
    group.add_argument(
        "--linear_use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AMP during linear probe feature extraction.",
    )
    group.add_argument(
        "--linear_probe_timestep",
        type=float,
        default=0.0,
        help="Fixed flow timestep t passed to student during linear probe.",
    )


def _create_npz_from_sample_folder(sample_dir: str, num: int, logger) -> str:
    samples = []
    for i in range(num):
        sample_path = os.path.join(sample_dir, f"{i:06d}.png")
        sample_pil = Image.open(sample_path)
        samples.append(np.asarray(sample_pil).astype(np.uint8))
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    logger.info("Saved sample npz to %s [shape=%s]", npz_path, samples.shape)
    return npz_path


def _compute_metrics_with_evaluator(args, sample_npz_path: str, logger) -> Dict[str, float]:
    if not args.fid_ref_path:
        logger.warning("FID evaluation skipped: --fid_ref_path not set.")
        return {}
    if not args.evaluator_script:
        logger.warning("FID evaluation skipped: --evaluator_script not set.")
        return {}

    eval_cwd = evaluator_working_dir(args.evaluator_script)
    evaluator_python = os.environ.get("EVALUATOR_PYTHON", sys.executable)
    cmd = [
        evaluator_python,
        args.evaluator_script,
        args.fid_ref_path,
        os.path.abspath(sample_npz_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=eval_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout.strip()
    if proc.returncode != 0:
        logger.error("Evaluator command failed (%d). Output:\n%s", proc.returncode, output)
        return {}
    logger.info("evaluator.py output:\n%s", output)

    metrics: Dict[str, float] = {}
    patterns = {
        "inception_score": r"Inception Score:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        "fid": r"FID:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        "sfid": r"sFID:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        "precision": r"Precision:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        "recall": r"Recall:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def _append_eval_metrics_record(metrics_all_path: str, record: dict, logger) -> None:
    history = []
    if os.path.isfile(metrics_all_path):
        try:
            with open(metrics_all_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                history = loaded
        except Exception as exc:
            logger.warning("Failed to read %s (%s); recreating.", metrics_all_path, exc)

    history = [
        x for x in history
        if not (isinstance(x, dict) and x.get("epoch") == record.get("epoch"))
    ]
    history.append(record)
    history.sort(key=lambda x: x.get("epoch", 0))

    tmp_path = metrics_all_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, metrics_all_path)
    logger.info("Updated eval summary at %s", metrics_all_path)


def reset_denoiser_training_state(model, model_without_ddp, optimizer, logger=None) -> None:
    """Restore train mode and clear stale grads after eval/linear-probe side paths."""
    model.train(True)
    model_without_ddp.train(True)
    optimizer.zero_grad(set_to_none=True)
    for param in model_without_ddp.parameters():
        if param.grad is not None:
            param.grad = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if misc.is_dist_avail_and_initialized():
        dist.barrier()
    if logger is not None and misc.is_main_process():
        logger.info("Reset denoiser training state after eval")


def merge_epoch_eval_metrics(results_dir: str, epoch: int, extra_fields: dict, logger) -> dict:
    eval_root = os.path.join(results_dir, "eval")
    stage_name = f"epoch_{epoch + 1:04d}"
    metric_path = os.path.join(eval_root, stage_name, "metrics.json")
    metrics_all_path = os.path.join(eval_root, "metrics_all.json")

    record = {"epoch": epoch + 1}
    if os.path.isfile(metric_path):
        try:
            with open(metric_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                record.update(loaded)
        except Exception as exc:
            logger.warning("Failed to read %s (%s); creating a new record.", metric_path, exc)

    record.update(extra_fields)
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    logger.info("Saved eval metrics to %s", metric_path)
    _append_eval_metrics_record(metrics_all_path, record, logger)
    return record


def _get_epoch_eval_record(results_dir: str, display_epoch: int):
    metrics_all_path = os.path.join(results_dir, "eval", "metrics_all.json")
    if not os.path.isfile(metrics_all_path):
        return None
    try:
        with open(metrics_all_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return None
    if not isinstance(history, list):
        return None
    for record in history:
        if isinstance(record, dict) and record.get("epoch") == display_epoch:
            return record
    return None


def _build_linear_probe_args(args):
    world_size = misc.get_world_size()
    per_gpu_batch_size = max(1, args.linear_global_batch_size // world_size)
    return argparse.Namespace(
        image_size=args.img_size,
        num_classes=args.class_num,
        train_data_path=args.linear_train_data_path,
        val_data_path=args.linear_val_data_path,
        train_label_file=args.linear_train_label_file,
        val_label_file=args.linear_val_label_file,
        epochs=args.linear_epochs,
        batch_size=per_gpu_batch_size,
        lr=args.linear_lr,
        num_workers=args.linear_num_workers,
        use_amp=args.linear_use_amp,
        pin_mem=True,
    )


def _restore_module_state_dict_inplace(module, saved_state_dict) -> None:
    """Restore weights without load_state_dict (preserves DDP Parameter identity)."""
    for name, param in module.named_parameters():
        if name in saved_state_dict:
            param.data.copy_(
                saved_state_dict[name].to(device=param.device, dtype=param.dtype)
            )
    for name, buffer in module.named_buffers():
        if name in saved_state_dict:
            buffer.data.copy_(
                saved_state_dict[name].to(device=buffer.device, dtype=buffer.dtype)
            )


def _switch_to_ema(model_without_ddp):
    saved_state_dict = {
        k: v.detach().clone() for k, v in model_without_ddp.state_dict().items()
    }
    for i, param in enumerate(model_without_ddp.parameters()):
        param.data.copy_(
            model_without_ddp.ema_params[i].to(device=param.device, dtype=param.dtype)
        )
    return saved_state_dict


def _restore_from_saved_state(model_without_ddp, saved_state_dict) -> None:
    _restore_module_state_dict_inplace(model_without_ddp, saved_state_dict)


def sample_images_for_fid(
    model_without_ddp,
    args,
    sample_dir: str,
    epoch: int,
    logger,
    *,
    num_sampling_steps: int | None = None,
) -> int:
    """Distributed image sampling; PNG names {index:06d}.png (SiT layout).

    Returns the number of sampling steps actually used.
    """
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    rank = misc.get_rank()
    batch_size = args.eval_per_proc_batch_size or args.gen_bsz

    global_batch_size = batch_size * world_size
    total_samples = int(
        np.ceil(args.eval_num_fid_samples / global_batch_size) * global_batch_size
    )
    samples_needed_this_gpu = total_samples // world_size
    iterations = samples_needed_this_gpu // batch_size

    saved_steps = model_without_ddp.steps
    steps_used = saved_steps if num_sampling_steps is None else num_sampling_steps
    if num_sampling_steps is not None:
        model_without_ddp.steps = num_sampling_steps

    if rank == 0:
        os.makedirs(sample_dir, exist_ok=True)
        logger.info(
            "FID eval epoch=%d: sampling %d images (target=%d) steps=%d cfg=%.2f -> %s",
            epoch + 1,
            total_samples,
            args.eval_num_fid_samples,
            steps_used,
            getattr(model_without_ddp, "cfg_scale", args.cfg),
            sample_dir,
        )
    if misc.is_dist_avail_and_initialized():
        dist.barrier()

    class_num = args.class_num
    uncondition = getattr(args, "uncondition", False)
    if uncondition:
        if rank == 0:
            logger.info(
                "FID eval epoch=%d: unconditional sampling (null class=%d)",
                epoch + 1,
                class_num,
            )
    else:
        assert args.eval_num_fid_samples % class_num == 0
    labels_per_class = args.eval_num_fid_samples // class_num if not uncondition else 0
    class_label_gen_world = np.arange(class_num).repeat(labels_per_class) if not uncondition else None
    if not uncondition:
        class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000, dtype=np.int64)])

    try:
        for it in range(iterations):
            start_idx = it * global_batch_size + rank * batch_size
            end_idx = start_idx + batch_size
            if uncondition:
                labels_gen = torch.full((batch_size,), class_num, device="cuda", dtype=torch.long)
            else:
                labels_gen = torch.from_numpy(class_label_gen_world[start_idx:end_idx]).long().cuda()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sampled_images = model_without_ddp.generate(labels_gen)

            if misc.is_dist_avail_and_initialized():
                dist.barrier()

            sampled_images = (sampled_images + 1) / 2
            sampled_images = sampled_images.detach().cpu()

            for b_id in range(sampled_images.size(0)):
                global_index = it * global_batch_size + rank * batch_size + b_id
                if global_index >= args.eval_num_fid_samples:
                    break
                gen_img = np.round(
                    np.clip(sampled_images[b_id].numpy().transpose(1, 2, 0) * 255, 0, 255)
                ).astype(np.uint8)
                Image.fromarray(gen_img).save(
                    os.path.join(sample_dir, f"{global_index:06d}.png")
                )

            if (it + 1) % max(1, args.eval_log_interval) == 0 and rank == 0:
                logger.info("FID eval sampling progress: %d/%d", it + 1, iterations)
    finally:
        model_without_ddp.steps = saved_steps

    if misc.is_dist_avail_and_initialized():
        dist.barrier()
    return steps_used


def _broadcast_bool(value: bool, src: int = 0) -> bool:
    if not misc.is_dist_avail_and_initialized():
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return bool(payload[0])


def evaluate_fid_after_epoch(
    model_without_ddp,
    args,
    epoch: int,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> None:
    """SiT-style eval: sample under /tmp, metrics under results_dir/eval/."""
    if not force and not args.eval_during_train:
        return
    if not force and not should_trigger_epoch_interval(args, epoch):
        return

    if logger is None:
        logger = logging.getLogger("jit_distill")

    rank = misc.get_rank()
    if not args.fid_ref_path or not args.evaluator_script:
        if rank == 0:
            logger.warning(
                "Skip periodic FID eval: set --fid_ref_path and --evaluator_script."
            )
        return

    results_dir = args.output_dir
    experiment_name = os.path.basename(results_dir.rstrip("/"))
    stage_name = f"epoch_{epoch + 1:04d}"
    eval_root = os.path.join(results_dir, "eval")
    stage_dir = os.path.join(eval_root, stage_name)
    sample_dir = os.path.join(
        args.eval_temp_root,
        experiment_name,
        stage_name,
        "samples",
    )
    metric_path = os.path.join(stage_dir, "metrics.json")
    metrics_all_path = os.path.join(eval_root, "metrics_all.json")

    skip_eval = False
    if rank == 0:
        missing = check_offline_eval_assets(args)
        if missing:
            logger.error(
                "FID eval skipped: offline SiT assets missing:\n  - %s",
                "\n  - ".join(missing),
            )
            skip_eval = True
        else:
            os.makedirs(eval_root, exist_ok=True)
            os.makedirs(stage_dir, exist_ok=True)
            if os.path.isdir(sample_dir):
                shutil.rmtree(sample_dir, ignore_errors=True)
    skip_eval = _broadcast_bool(skip_eval, src=0)
    if skip_eval:
        if rank == 0:
            logger.warning("FID eval aborted for epoch %d (missing assets)", epoch + 1)
        return

    if misc.is_dist_avail_and_initialized():
        dist.barrier()

    fd_eval_steps = getattr(args, "fd_eval_num_sampling_steps", None)
    model_state = _switch_to_ema(model_without_ddp)
    try:
        steps_used = sample_images_for_fid(
            model_without_ddp,
            args,
            sample_dir,
            epoch,
            logger,
            num_sampling_steps=fd_eval_steps,
        )
    finally:
        _restore_from_saved_state(model_without_ddp, model_state)

    if misc.is_dist_avail_and_initialized():
        dist.barrier()

    postprocess_ok = True
    fdr6_metrics: dict[str, Any] = {}
    ran_fdr6 = False
    if rank == 0:
        logger.info(
            "FID eval epoch=%d: sampling finished; rank0 building NPZ and running evaluator "
            "(other ranks waiting at barrier)",
            epoch + 1,
        )
        try:
            sample_npz_path = _create_npz_from_sample_folder(
                sample_dir, args.eval_num_fid_samples, logger
            )
            eval_metrics = _compute_metrics_with_evaluator(args, sample_npz_path, logger)
            release_eval_cuda_memory(logger)

            if getattr(args, "eval_fdr6", False):
                skip_fdr6 = False
                missing_fdr6 = check_fdr6_assets(args)
                if missing_fdr6:
                    logger.warning(
                        "FDr-6 skipped for epoch %d (missing assets):\n  - %s",
                        epoch + 1,
                        "\n  - ".join(missing_fdr6),
                    )
                    skip_fdr6 = True
                else:
                    logger.info(
                        "FID eval epoch=%d: running FDr-6 on %d samples in %s (rank0 only)",
                        epoch + 1,
                        args.eval_num_fid_samples,
                        sample_dir,
                    )
                if not skip_fdr6:
                    try:
                        fdr6_metrics = evaluate_fdr6_from_folder(
                            sample_dir, args, logger_obj=logger
                        )
                        ran_fdr6 = True
                    except Exception:
                        logger.exception(
                            "FDr-6 eval failed for epoch %d (training continues)", epoch + 1
                        )
                        fdr6_metrics = {}
                release_eval_cuda_memory(logger, reset_compile=ran_fdr6)

            remove_temporary = not args.keep_eval_samples
            record = {
                "epoch": epoch + 1,
                "num_fid_samples": args.eval_num_fid_samples,
                "fid_ref_path": args.fid_ref_path,
                "cfg": args.cfg,
                "num_sampling_steps": steps_used,
                "sample_npz_path": None if remove_temporary else sample_npz_path,
                "temporary_files_removed": remove_temporary,
                **eval_metrics,
                **fdr6_metrics,
            }
            with open(metric_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            logger.info("Saved eval metrics to %s", metric_path)
            _append_eval_metrics_record(metrics_all_path, record, logger)

            if remove_temporary:
                shutil.rmtree(sample_dir, ignore_errors=True)
                if os.path.isfile(sample_npz_path):
                    os.remove(sample_npz_path)
                logger.info("Removed temporary eval files under %s", args.eval_temp_root)
        except Exception:
            postprocess_ok = False
            logger.exception(
                "FID eval post-processing failed for epoch %d (NPZ/metrics/evaluator)",
                epoch + 1,
            )

    if misc.is_dist_avail_and_initialized():
        dist.barrier()

    postprocess_ok = _broadcast_bool(postprocess_ok, src=0)
    ran_fdr6 = _broadcast_bool(ran_fdr6, src=0)

    release_eval_cuda_memory(logger if rank == 0 else None, reset_compile=ran_fdr6)

    if rank == 0:
        if postprocess_ok:
            logger.info("FID eval epoch=%d: all ranks synced; returning to training", epoch + 1)
        else:
            logger.error(
                "FID eval epoch=%d: post-processing failed; all ranks synced, returning to training",
                epoch + 1,
            )
    if not postprocess_ok:
        raise RuntimeError(f"FID eval post-processing failed for epoch {epoch + 1}")


def evaluate_linear_probe_after_epoch(
    model,
    model_without_ddp,
    optimizer,
    args,
    epoch: int,
    train_steps: int,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
    student_ema=None,
) -> None:
    if not force and not getattr(args, "linear_eval_during_train", True):
        return
    if not getattr(args, "uncondition", False):
        return
    if getattr(args, "distill_loss_weight", 0.0) <= 0:
        return
    if not hasattr(model_without_ddp, "net_student"):
        return
    if not force and not should_trigger_linear_eval(args, epoch):
        return

    if logger is None:
        logger = logging.getLogger("jit_distill")

    rank = misc.get_rank()
    if not args.linear_train_data_path or not args.linear_val_data_path:
        if rank == 0:
            logger.warning(
                "Skip linear probe eval: set --linear_train_data_path and --linear_val_data_path."
            )
        return

    probe_student = student_ema if student_ema is not None else model_without_ddp.net_student
    weight_source = "student_ema" if student_ema is not None else "net_student"
    if rank == 0:
        logger.info(
            "Linear probe eval epoch=%d: training linear head on frozen %s features",
            epoch + 1,
            weight_source,
        )

    from linear_jit_student import run_linear_probe

    student_was_training = probe_student.training
    probe_student.eval()
    if student_ema is not None:
        sync_student_ema_from_main_ema(model_without_ddp, student_ema)
    try:
        linear_args = _build_linear_probe_args(args)
        result = run_linear_probe(
            linear_args,
            device=torch.device(
                f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
            ),
            local_rank=rank,
            student=probe_student,
            use_ddp=False,
        )
    finally:
        if student_was_training:
            probe_student.train()

    if misc.is_dist_avail_and_initialized():
        dist.barrier()

    if rank == 0:
        merge_epoch_eval_metrics(
            args.output_dir,
            epoch,
            {
                "train_steps": train_steps,
                "linear_best_acc": result["best_acc"],
                "linear_final_acc": result["final_acc"],
                "linear_epochs": args.linear_epochs,
                "linear_global_batch_size": args.linear_global_batch_size,
                "linear_train_data_path": args.linear_train_data_path,
                "linear_val_data_path": args.linear_val_data_path,
                "linear_train_label_file": args.linear_train_label_file,
                "linear_val_label_file": args.linear_val_label_file,
            },
            logger,
        )

    if misc.is_dist_avail_and_initialized():
        dist.barrier()


def maybe_run_pending_linear_eval_on_resume(
    model,
    model_without_ddp,
    optimizer,
    args,
    resume_state: dict,
    train_steps: int,
    logger,
    student_ema=None,
) -> None:
    if not resume_state.get("loaded"):
        return
    if not getattr(args, "linear_eval_during_train", True):
        return
    if not getattr(args, "uncondition", False):
        return
    if getattr(args, "distill_loss_weight", 0.0) <= 0:
        return

    display_epoch = resume_state["start_epoch"]
    if display_epoch <= 0:
        return
    if display_epoch < getattr(args, "linear_eval_start_epoch", 1):
        return
    if not should_trigger_epoch_interval(args, display_epoch - 1):
        return

    record = _get_epoch_eval_record(args.output_dir, display_epoch)
    if record is not None and "linear_best_acc" in record:
        return

    if misc.is_main_process():
        logger.info(
            "Resume at epoch %d: running pending linear probe eval before continuing training",
            display_epoch,
        )
    evaluate_linear_probe_after_epoch(
        model,
        model_without_ddp,
        optimizer,
        args,
        display_epoch - 1,
        train_steps,
        logger=logger,
        force=True,
        student_ema=student_ema,
    )
