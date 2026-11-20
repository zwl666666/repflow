"""JiT FD post-training: optimize generator with student-repr Fréchet loss."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import util.misc as misc
from engine_fd import adjust_lr, get_fd_train_step
from eval_utils import add_eval_args, evaluate_fid_after_epoch, release_eval_cuda_memory
from fdr6_eval import check_fdr6_assets
from frechet_distance.judges_jit import (
    build_student_judge,
    fill_all_queues,
    load_fd_queue_states,
    run_sanity_check,
    save_fd_queue_states,
)
from frechet_distance.student_repr import build_student_repr_model
from train_distill import (
    add_architecture_args,
    add_data_args,
    add_flow_args,
    add_optimization_args,
    add_runtime_args,
    add_sampling_args,
)
from train_utils import (
    add_distill_training_args,
    add_model_structure_args,
    apply_ema_weights_from_checkpoint,
    apply_eval_defaults,
    build_denoiser,
    compute_per_device_batch_size,
    create_logger,
    init_ema_params,
    install_exception_logging,
    load_denoiser_state_dict,
    denoiser_load_exclude_prefixes,
    log_model_summary,
    log_offline_eval_config,
    resolve_checkpoint_path,
    setup_experiment_directory,
    setup_training_paths,
)

logger = logging.getLogger("jit_fd")


def add_fd_args(parser):
    group = parser.add_argument_group("FD post-training")
    group.add_argument("--load_from", required=True, type=str, help="Distillation checkpoint to initialize from")
    group.add_argument("--args_json", default=None, type=str, help="Architecture args.json (default: sibling of load_from)")
    group.add_argument("--fd_ref_stats", required=True, type=str, help="Student repr reference stats (.npz)")
    group.add_argument("--queue_size", type=int, default=50000)
    group.add_argument("--fd_fid_norm_eps", type=float, default=0.01)
    group.add_argument("--fd_queue_fill_bsz", type=int, default=256)
    group.add_argument("--fd_online_accum", action="store_true")
    group.add_argument("--fd_eigvalsh", action="store_true")
    group.add_argument("--fd_ema_beta", type=float, default=0.999)
    group.add_argument("--fd_num_sampling_steps", type=int, default=1)
    group.add_argument(
        "--fd_eval_num_sampling_steps",
        type=int,
        default=1,
        help="Sampling steps for periodic FID eval during FD post-training (default: 1, aligned with train).",
    )
    group.add_argument(
        "--fd_train_cfg",
        type=float,
        default=1.0,
        help="CFG for differentiable FD training step (default: 1.0).",
    )
    group.add_argument(
        "--fd_eval_cfg",
        type=float,
        default=None,
        help="CFG for queue fill and periodic FID eval (default: same as --fd_train_cfg).",
    )
    group.add_argument("--fd_student_t", type=float, default=1.0)
    group.add_argument("--fd_use_projector", action="store_true")
    group.add_argument("--fd_student_use_mask", action="store_true")
    group.add_argument(
        "--no_fd_student_apply_norm",
        action="store_false",
        dest="fd_student_apply_norm",
        help="Skip student final LayerNorm in FD feature extraction (default: apply norm without mask).",
    )
    parser.set_defaults(fd_student_apply_norm=True)
    group.add_argument("--fd_student_pool", default="mean", choices=["mean", "sum", "flatten"])
    group.add_argument(
        "--fd_use_ema",
        action="store_true",
        help="Load net + net_student from checkpoint model_ema (generator and FD judge).",
    )
    group.add_argument("--fd_loss_weight", type=float, default=1.0)
    group.add_argument("--total_steps", type=int, default=125000)
    group.add_argument("--fd_warmup_steps", type=int, default=6250)
    group.add_argument("--print_freq", type=int, default=50)
    group.add_argument("--save_every", type=int, default=5000)
    group.add_argument("--eval_every", type=int, default=10000)
    group.add_argument("--grad_clip", type=float, default=0.0)
    group.add_argument(
        "--eval_at_start",
        action="store_true",
        help="Run one FID eval after resume/queue init, before the training loop.",
    )


def get_args_parser():
    parser = argparse.ArgumentParser("JiT FD post-training", add_help=True)
    add_architecture_args(parser)
    add_model_structure_args(parser)
    add_optimization_args(parser)
    add_flow_args(parser)
    add_sampling_args(parser)
    add_data_args(parser)
    add_distill_training_args(parser)
    add_eval_args(parser)
    add_runtime_args(parser)
    add_fd_args(parser)
    parser.set_defaults(eval_fdr6=True)
    return parser


def load_architecture_args(args_json: str) -> argparse.Namespace:
    with open(args_json, encoding="utf-8") as f:
        payload = json.load(f)
    return argparse.Namespace(**payload)


def merge_architecture_into(args, arch: argparse.Namespace):
    arch_keys = [
        "training_mode", "model", "student_model", "img_size",
        "attn_dropout", "proj_dropout", "teacher_feature_layers", "teacher_cons_layer",
        "student_mask_ratio", "projector_model", "projector_block_dim",
        "projector_num_blocks", "projector_num_heads", "projector_mlp_ratio",
        "projector_out_dim", "projector_attn_dropout", "projector_proj_dropout",
        "projector_num_layers", "projector_feature_layers", "projector_input_hidden_size",
        "P_mean", "P_std", "noise_scale", "t_eps", "label_drop_prob",
        "class_num", "sampling_method", "gen_bsz",
    ]
    for key in arch_keys:
        if hasattr(arch, key):
            setattr(args, key, getattr(arch, key))
    args.training_mode = "distill"


def resolve_fd_eval_cfg(args) -> float:
    if args.fd_eval_cfg is not None:
        return args.fd_eval_cfg
    return args.fd_train_cfg


def apply_denoiser_sampling_config(denoiser, args, log=None):
    """Sync denoiser.generate() cfg/interval for queue fill and periodic FID eval."""
    eval_cfg = resolve_fd_eval_cfg(args)
    denoiser.cfg_scale = eval_cfg
    denoiser.cfg_interval = (args.interval_min, args.interval_max)
    denoiser.steps = args.num_sampling_steps
    args.cfg = eval_cfg
    if log is not None:
        stats_mode = (
            f"ema(beta={args.fd_ema_beta})"
            if args.fd_ema_beta > 0
            else ("online_accum" if args.fd_online_accum else "queue_snapshot")
        )
        log.info(
            "FD sampling: train fd_steps=%d fd_train_cfg=%.2f | "
            "eval fd_eval_steps=%d fd_eval_cfg=%.2f | queue fill fd_steps=%d | queue_stats=%s",
            args.fd_num_sampling_steps,
            args.fd_train_cfg,
            getattr(args, "fd_eval_num_sampling_steps", args.fd_num_sampling_steps),
            eval_cfg,
            args.fd_num_sampling_steps,
            stats_mode,
        )


def freeze_non_generator(denoiser):
    for p in denoiser.net_student.parameters():
        p.requires_grad_(False)
    if getattr(denoiser, "projector", None) is not None:
        for p in denoiser.projector.parameters():
            p.requires_grad_(False)
    denoiser.net_student.eval()
    if getattr(denoiser, "projector", None) is not None:
        denoiser.projector.eval()


def save_fd_checkpoint(args, step, denoiser, optimizer, judges):
    ckpt_root = Path(getattr(args, "checkpoint_dir", None) or args.output_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    to_save = {
        "model": denoiser.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": args,
        "fd_queue_states": save_fd_queue_states(judges),
    }
    ema_state_dict = copy.deepcopy(denoiser.state_dict())
    for i, (name, _value) in enumerate(denoiser.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = denoiser.ema_params[i]
    to_save["model_ema"] = ema_state_dict
    misc.save_on_master(to_save, ckpt_root / "checkpoint-latest.pth")


def load_initial_weights(denoiser, load_from: str, args, logger_obj):
    checkpoint = torch.load(load_from, map_location="cpu", weights_only=False)
    exclude = denoiser_load_exclude_prefixes(args)
    load_denoiser_state_dict(
        denoiser,
        checkpoint["model"],
        exclude_prefixes=exclude,
        logger=logger_obj if exclude else None,
        strict=not exclude,
    )
    if args.fd_use_ema:
        source = apply_ema_weights_from_checkpoint(
            denoiser,
            checkpoint,
            skip_projector=getattr(args, "skip_projector", False),
            logger=logger_obj,
        )
        logger_obj.info("FD init: generator (net) + judge student from EMA (source=%s)", source)
    else:
        logger_obj.info("FD init: online weights from checkpoint['model'] (net + net_student)")
    logger_obj.info("Loaded generator checkpoint from %s", load_from)
    del checkpoint


def load_fd_resume(resume_path, denoiser, optimizer, judges, ema_device, logger_obj, args):
    if not resume_path or not os.path.isfile(resume_path):
        return 0, False

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    exclude = denoiser_load_exclude_prefixes(args)
    load_denoiser_state_dict(
        denoiser,
        checkpoint["model"],
        exclude_prefixes=exclude,
        logger=logger_obj if exclude else None,
        strict=not exclude,
    )
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    ema_state_dict = checkpoint.get("model_ema", checkpoint.get("model_ema1"))
    if ema_state_dict is not None:
        denoiser.ema_params = [
            ema_state_dict[name].detach().to(ema_device).clone()
            for name, _ in denoiser.named_parameters()
        ]

    start_step = int(checkpoint.get("step", 0))
    restored = False
    if "fd_queue_states" in checkpoint:
        restored = load_fd_queue_states(judges, checkpoint["fd_queue_states"])

    logger_obj.info("Resumed FD checkpoint from %s (step=%d, queue_restored=%s)", resume_path, start_step, restored)
    del checkpoint
    return start_step, restored


def main(args):
    misc.init_distributed_mode(args)
    rank = misc.get_rank()

    if args.args_json is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.load_from))
        candidate = os.path.join(ckpt_dir, "args.json")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"args.json not found: {candidate}")
        args.args_json = candidate

    arch = load_architecture_args(args.args_json)
    merge_architecture_into(args, arch)

    experiment_dir = setup_experiment_directory(args, rank)
    results_dir, checkpoint_dir = setup_training_paths(args, rank)
    args.output_dir = results_dir
    args.checkpoint_dir = checkpoint_dir

    log = create_logger(results_dir, rank=rank)
    install_exception_logging(log, results_dir, rank)
    log.info("FD post-training job dir: %s", os.path.dirname(os.path.realpath(__file__)))
    log.info("Results dir: %s", results_dir)
    log.info("Arguments:\n%s", args)

    apply_eval_defaults(args)
    if rank == 0 and args.eval_during_train:
        log_offline_eval_config(log, args)
        if getattr(args, "eval_fdr6", False):
            missing_fdr6 = check_fdr6_assets(args)
            if missing_fdr6:
                log.warning(
                    "FDr-6 enabled but assets missing (will skip at eval time):\n  - %s",
                    "\n  - ".join(missing_fdr6),
                )
            else:
                log.info(
                    "FDr-6 eval enabled: fd_loss_root=%s repr_weights=%s batch=%d",
                    args.fdr6_fd_loss_root,
                    args.fdr6_repr_weights_root,
                    args.fdr6_batch_size,
                )

    device = torch.device(args.device)
    ema_device = torch.device("cpu") if args.ema_on_cpu else device
    cudnn.benchmark = True

    per_device_bs = compute_per_device_batch_size(args)
    args.batch_size = per_device_bs
    eff_batch_size = per_device_bs * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    log.info("Per-device batch=%d effective=%d lr=%.2e", per_device_bs, eff_batch_size, args.lr)

    args.skip_projector = not args.fd_use_projector
    if args.skip_projector and rank == 0:
        log.info("[FD] skip_projector=True (set USE_PROJECTOR=1 / --fd_use_projector to load projector)")
    denoiser = build_denoiser(args)
    log_model_summary(log, args, denoiser)
    load_initial_weights(denoiser, args.load_from, args, log)
    freeze_non_generator(denoiser)
    denoiser.to(device)
    denoiser.ema_params = init_ema_params(denoiser, ema_device)
    apply_denoiser_sampling_config(denoiser, args, log)

    repr_model = build_student_repr_model(
        denoiser,
        use_projector=args.fd_use_projector,
        fd_t=args.fd_student_t,
        use_mask=args.fd_student_use_mask,
        apply_final_norm=args.fd_student_apply_norm,
        pool=args.fd_student_pool,
    )

    if args.fd_ema_beta > 0:
        args.fd_online_accum = True

    judge_name = "ema" if args.fd_use_ema else "student"
    judges = [
        build_student_judge(
            repr_model,
            args.fd_ref_stats,
            queue_size=args.queue_size,
            fd_online_accum=args.fd_online_accum,
            fd_ema_beta=args.fd_ema_beta,
            fd_eigvalsh=args.fd_eigvalsh,
            weight=args.fd_loss_weight,
            name=judge_name,
        )
    ]

    trainable = [p for p in denoiser.net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    resume_path = None
    if args.resume:
        resume_path = resolve_checkpoint_path(args.resume, args.resume_checkpoint)
    start_step, queue_restored = load_fd_resume(
        resume_path, denoiser, optimizer, judges, ema_device, log, args,
    )
    if not queue_restored:
        log.info("[FD] Filling feature queue (%d entries) ...", args.queue_size)
        fill_all_queues(judges, denoiser, args)
        repr_model.to(device)
    else:
        log.info("[FD] Restored queue from checkpoint")
    run_sanity_check(judges, args.queue_size, args=args)

    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    denoiser.train()
    denoiser.net_student.eval()
    if getattr(denoiser, "projector", None) is not None:
        denoiser.projector.eval()
    repr_model.to(device)

    fd_train_step = get_fd_train_step(denoiser, judges, args)
    global_bsz = args.batch_size * misc.get_world_size()
    step_start = time.perf_counter()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    log.info("FD training: step %d -> %d", start_step, args.total_steps)

    if args.eval_at_start and args.eval_during_train:
        log.info("[FD] Running FID eval at start (step=%d) before training loop", start_step)
        torch.cuda.empty_cache()
        evaluate_fid_after_epoch(
            denoiser,
            args,
            start_step,
            logger=log,
            force=True,
        )
        denoiser.train()
        denoiser.net.train()
        denoiser.net_student.eval()
        if getattr(denoiser, "projector", None) is not None:
            denoiser.projector.eval()
        release_eval_cuda_memory(log, reset_compile=getattr(args, "eval_fdr6", False))
        gc.collect()

    for step in range(start_step, args.total_steps):
        denoiser.train()
        denoiser.net.train()
        adjust_lr(optimizer, step, args)

        loss, loss_dict = fd_train_step()

        grad_norm = 0.0
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        else:
            grad_norm = math.sqrt(sum(p.grad.detach().norm(2).item() ** 2 for p in trainable if p.grad is not None))

        if torch.isfinite(torch.tensor(grad_norm)):
            optimizer.step()
            denoiser.maybe_update_ema()
        else:
            log.warning("[step %d] NaN/Inf grad_norm — skipping update", step)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()

        loss_value = misc.all_reduce_mean(loss.item())
        loss_dict = {k: misc.all_reduce_mean(v) for k, v in loss_dict.items()}
        lr = optimizer.param_groups[0]["lr"]

        if step % args.print_freq == 0:
            sps = args.batch_size / step_time if step_time > 0 else 0.0
            fd_str = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            log.info(
                "step %d/%d loss=%.4f %s lr=%.2e grad=%.3f sps/dev=%.2f",
                step,
                args.total_steps,
                loss_value,
                fd_str,
                lr,
                grad_norm,
                sps,
            )

        if (step + 1) % args.save_every == 0 or step + 1 == args.total_steps:
            save_fd_checkpoint(args, step + 1, denoiser, optimizer, judges)

        if args.eval_during_train and (step + 1) % args.eval_every == 0:
            torch.cuda.empty_cache()
            evaluate_fid_after_epoch(
                denoiser,
                args,
                step + 1,
                logger=log,
                force=True,
            )
            denoiser.train()
            denoiser.net.train()
            denoiser.net_student.eval()
            if getattr(denoiser, "projector", None) is not None:
                denoiser.projector.eval()
            release_eval_cuda_memory(log, reset_compile=getattr(args, "eval_fdr6", False))
            gc.collect()

    log.info("FD post-training complete (%d steps, %d samples)", args.total_steps, args.total_steps * global_bsz)


if __name__ == "__main__":
    parser = get_args_parser()
    cli_args = parser.parse_args()
    cli_args.training_mode = "distill"
    main(cli_args)
