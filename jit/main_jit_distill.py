import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch._dynamo
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from util.crop import center_crop_arr
import util.misc as misc

from engine_jit_distill import train_one_epoch
from eval_utils import (
    add_eval_args,
    evaluate_fid_after_epoch,
    evaluate_linear_probe_after_epoch,
    maybe_run_pending_linear_eval_on_resume,
    reset_denoiser_training_state,
)
from train_utils import (
    add_distill_training_args,
    add_loss_weight_args,
    add_model_structure_args,
    add_repa_args,
    add_sra_args,
    add_training_mode_args,
    apply_eval_defaults,
    apply_loss_weight_aliases,
    build_denoiser,
    build_student_ema,
    compute_per_device_batch_size,
    create_logger,
    get_effective_resume,
    init_ema_params,
    install_exception_logging,
    is_baseline_training,
    load_checkpoint,
    log_model_summary,
    log_offline_eval_config,
    maybe_update_student_ema,
    resolve_checkpoint_path,
    save_model_and_sync,
    setup_experiment_directory,
    setup_training_paths,
    should_trigger_epoch_interval,
    sync_student_ema_from_main_ema,
)


def add_architecture_args(parser):
    group = parser.add_argument_group("Architecture")
    group.add_argument("--model", default="JiT-B/16", type=str, metavar="MODEL")
    group.add_argument("--student_model", default="JiT-B/16", type=str, metavar="MODEL")
    group.add_argument("--img_size", default=256, type=int)
    group.add_argument("--attn_dropout", type=float, default=0.0)
    group.add_argument("--proj_dropout", type=float, default=0.0)


def add_optimization_args(parser):
    group = parser.add_argument_group("Optimization")
    group.add_argument("--epochs", default=200, type=int)
    group.add_argument("--warmup_epochs", type=int, default=5, metavar="N")
    group.add_argument("--batch_size", default=128, type=int)
    group.add_argument("--lr", type=float, default=None, metavar="LR")
    group.add_argument("--blr", type=float, default=5e-5, metavar="LR")
    group.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    group.add_argument("--lr_schedule", type=str, default="constant")
    group.add_argument("--weight_decay", type=float, default=0.0)
    group.add_argument("--ema_decay", type=float, default=0.9999)
    group.add_argument(
        "--ema_update_freq",
        type=int,
        default=1,
        metavar="N",
        help="Update EMA every N optimizer steps (decay is adjusted so smoothing rate is similar).",
    )
    group.add_argument(
        "--ema_on_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store EMA shadow weights on CPU (slower; default keeps EMA on GPU).",
    )


def add_flow_args(parser):
    group = parser.add_argument_group("Flow matching")
    group.add_argument("--P_mean", default=-0.8, type=float)
    group.add_argument("--P_std", default=0.8, type=float)
    group.add_argument("--noise_scale", default=1.0, type=float)
    group.add_argument("--t_eps", default=5e-2, type=float)
    group.add_argument("--label_drop_prob", default=0.1, type=float)
    group.add_argument(
        "--uncondition",
        action="store_true",
        help="Train unconditionally (always use null class label). Enables linear probe eval.",
    )


def add_sampling_args(parser):
    group = parser.add_argument_group("Sampling")
    group.add_argument("--sampling_method", default="heun", type=str)
    group.add_argument("--num_sampling_steps", default=50, type=int)
    group.add_argument(
        "--cfg",
        default=1.0,
        type=float,
        help="Classifier-free guidance scale; <=1 disables CFG during sampling.",
    )
    group.add_argument("--interval_min", default=0.0, type=float)
    group.add_argument("--interval_max", default=1.0, type=float)
    group.add_argument("--gen_bsz", type=int, default=128)
    group.add_argument("--evaluate_gen", action="store_true")


def add_data_args(parser):
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--data_path",
        default="./data/imagenet",
        type=str,
        help="Local ImageNet train root (CLS-LOC/train, ImageFolder layout).",
    )
    group.add_argument("--class_num", default=1000, type=int)
    group.add_argument("--num_workers", default=12, type=int)
    group.add_argument(
        "--pin_mem",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def add_runtime_args(parser):
    group = parser.add_argument_group("Runtime")
    group.add_argument("--seed", default=0, type=int)
    group.add_argument("--start_epoch", default=0, type=int, metavar="N")
    group.add_argument("--log_freq", default=100, type=int)
    group.add_argument("--device", default="cuda")
    group.add_argument(
        "--output_dir",
        default="./output_dir",
        help="Persistent results directory (logs, eval metrics).",
    )
    group.add_argument("--save_last_freq", type=int, default=0)
    group.add_argument("--save_freq", type=int, default=100)
    group.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="Number of distributed processes (set by torchrun).",
    )
    group.add_argument("--local_rank", default=-1, type=int)
    group.add_argument("--dist_on_itp", action="store_true")
    group.add_argument("--dist_url", default="env://")


def get_args_parser():
    parser = argparse.ArgumentParser("JiT distillation", add_help=True)
    add_training_mode_args(parser)
    add_architecture_args(parser)
    add_model_structure_args(parser)
    add_loss_weight_args(parser)
    add_repa_args(parser)
    add_sra_args(parser)
    add_optimization_args(parser)
    add_flow_args(parser)
    add_sampling_args(parser)
    add_eval_args(parser)
    add_data_args(parser)
    add_distill_training_args(parser)
    add_runtime_args(parser)
    return parser


def build_dataloader(args, transform_train):
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"Local --data_path not found: {args.data_path}")
    from torchvision.datasets import ImageFolder
    dataset_train = ImageFolder(args.data_path, transform_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=misc.get_world_size(),
        rank=misc.get_rank(),
        shuffle=True,
    )
    per_device_bs = compute_per_device_batch_size(args)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=per_device_bs,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    return dataset_train, sampler_train, data_loader_train, per_device_bs


def save_training_args(args, experiment_dir, rank):
    if rank != 0:
        return
    args_path = os.path.join(experiment_dir, "args.json")
    payload = {
        k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
        for k, v in vars(args).items()
    }
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _resolve_resume_path(args):
    resume_raw = get_effective_resume(args)
    if not resume_raw:
        return None
    resume_raw = os.path.expanduser(resume_raw)
    if os.path.isdir(resume_raw) and os.path.abspath(resume_raw) == os.path.abspath(args.output_dir):
        resume_raw = args.checkpoint_dir
    return resolve_checkpoint_path(resume_raw, args.resume_checkpoint)


def main(args):
    misc.init_distributed_mode(args)
    rank = misc.get_rank()

    experiment_dir = setup_experiment_directory(args, rank)
    results_dir, checkpoint_dir = setup_training_paths(args, rank)
    args.output_dir = results_dir
    args.checkpoint_dir = checkpoint_dir

    logger = create_logger(results_dir, rank=rank)
    install_exception_logging(logger, results_dir, rank)
    logger.info("Job directory: %s", os.path.dirname(os.path.realpath(__file__)))
    logger.info("Results dir: %s", results_dir)
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("Arguments:\n%s", args)
    if rank == 0 and args.eval_during_train:
        log_offline_eval_config(logger, args)

    device = torch.device(args.device)

    base_seed = args.global_seed if args.global_seed is not None else args.seed
    seed = base_seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=results_dir)
        save_training_args(args, results_dir, rank)
    else:
        log_writer = None

    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])

    dataset_train, sampler_train, data_loader_train, per_device_bs = build_dataloader(
        args, transform_train
    )
    args.batch_size = per_device_bs
    if args.eval_per_proc_batch_size is None:
        args.eval_per_proc_batch_size = args.gen_bsz
    args.num_images = args.eval_num_fid_samples

    logger.info("Dataset: %s (%d images)", dataset_train, len(dataset_train))
    logger.info("Per-device batch size: %d", per_device_bs)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    apply_loss_weight_aliases(args)
    sra_w = float(getattr(args, "sra_loss_weight", 0.0) or 0.0)
    if sra_w > 0 and not is_baseline_training(args):
        raise ValueError("SRA requires --training_mode baseline (no student).")
    if is_baseline_training(args):
        args.distill_loss_weight = 0.0
        repa_w = float(getattr(args, "repa_loss_weight", 0.0) or 0.0)
        if sra_w > 0:
            logger.info(
                "training_mode=baseline+SRA: teacher flow + self-representation align "
                "(no student/projector; sra_loss_weight=%.4f, s=%s, t=%s)",
                sra_w,
                getattr(args, "sra_block_out_s", 8),
                getattr(args, "sra_block_out_t", 20),
            )
        elif repa_w > 0:
            logger.info(
                "training_mode=baseline+REPA: teacher flow + DINOv2 align "
                "(no student/projector; repa_loss_weight=%.4f)",
                repa_w,
            )
        else:
            logger.info("training_mode=baseline: teacher flow loss only (no student/projector)")
    model = build_denoiser(args)
    log_model_summary(logger, args, model)

    model.to(device)
    ema_device = torch.device("cpu") if args.ema_on_cpu else device

    eff_batch_size = per_device_bs * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    logger.info("Actual lr: %.2e | effective batch size: %d", args.lr, eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    student_ema = None
    if getattr(args, "distill_loss_weight", 0.0) > 0 and hasattr(model_without_ddp, "net_student"):
        student_ema = build_student_ema(model_without_ddp, device)
        logger.info("Initialized standalone student_ema for linear probe (SiT-style)")

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    checkpoint_path = _resolve_resume_path(args)
    resume_state = load_checkpoint(
        checkpoint_path,
        model_without_ddp,
        optimizer,
        ema_device,
        logger=logger,
        student_ema=student_ema,
    )
    if resume_state["loaded"]:
        args.start_epoch = resume_state["start_epoch"]
    else:
        model_without_ddp.ema_params = init_ema_params(model_without_ddp, ema_device)
        if student_ema is not None:
            maybe_update_student_ema(student_ema, model_without_ddp, force=True)
        logger.info("Training from scratch")

    logger.info(
        "EMA: device=%s decay=%.6f update_freq=%d",
        ema_device,
        args.ema_decay,
        args.ema_update_freq,
    )

    if args.evaluate_gen:
        logger.info("Running one-shot FID evaluation (cfg=%.2f)", args.cfg)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate_fid_after_epoch(
                    model_without_ddp,
                    args,
                    max(args.start_epoch - 1, 0),
                    logger=logger,
                    force=True,
                )
        return

    logger.info("Start training for %d epochs (from epoch %d)", args.epochs, args.start_epoch)
    start_time = time.time()
    steps_per_epoch = len(data_loader_train)
    train_steps = args.start_epoch * steps_per_epoch

    maybe_run_pending_linear_eval_on_resume(
        model,
        model_without_ddp,
        optimizer,
        args,
        resume_state,
        train_steps,
        logger,
        student_ema=student_ema,
    )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        epoch_wall_start = time.time()
        _, train_steps = train_one_epoch(
            model,
            model_without_ddp,
            data_loader_train,
            optimizer,
            device,
            epoch,
            log_writer=log_writer,
            args=args,
            logger=logger,
            train_steps_offset=train_steps,
            student_ema=student_ema,
        )
        model_without_ddp.maybe_update_ema(force=True)
        maybe_update_student_ema(student_ema, model_without_ddp, force=True)

        if misc.is_main_process():
            epoch_wall = time.time() - epoch_wall_start
            logger.info(
                "Epoch %d train loop done in %s (global_step=%d)",
                epoch,
                str(datetime.timedelta(seconds=int(epoch_wall))),
                train_steps,
            )

        is_last_epoch = epoch + 1 == args.epochs
        if is_last_epoch:
            ckpt_start = time.time()
            save_model_and_sync(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                logger=logger,
                student_ema=student_ema,
            )
            logger.info(
                "Saved checkpoint-latest (epoch=%d) to %s (%.1fs)",
                epoch,
                args.checkpoint_dir,
                time.time() - ckpt_start,
            )

        if should_trigger_epoch_interval(args, epoch) and epoch > 0:
            ckpt_start = time.time()
            save_model_and_sync(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                logger=logger,
                student_ema=student_ema,
            )
            logger.info(
                "Saved checkpoint-latest (epoch=%d) to %s (%.1fs)",
                epoch,
                args.checkpoint_dir,
                time.time() - ckpt_start,
            )

        if args.save_last_every_epoch and not is_last_epoch:
            ckpt_start = time.time()
            save_model_and_sync(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                logger=logger,
                student_ema=student_ema,
            )
            logger.info(
                "Saved checkpoint-latest (epoch=%d) to %s (%.1fs)",
                epoch,
                args.checkpoint_dir,
                time.time() - ckpt_start,
            )

        if should_trigger_epoch_interval(args, epoch):
            logger.info("Starting FID eval after epoch %d (global_step=%d)", epoch + 1, train_steps)
            eval_start = time.time()
            torch.cuda.empty_cache()
            try:
                with torch.no_grad():
                    evaluate_fid_after_epoch(
                        model_without_ddp, args, epoch, logger=logger
                    )
            except Exception:
                logger.exception(
                    "FID eval failed after epoch %d; continuing training",
                    epoch + 1,
                )
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model.train(True)
            logger.info(
                "FID eval after epoch %d finished in %s; resuming training (next epoch=%d)",
                epoch + 1,
                str(datetime.timedelta(seconds=int(time.time() - eval_start))),
                epoch + 1,
            )

        if should_trigger_epoch_interval(args, epoch):
            linear_start = time.time()
            try:
                evaluate_linear_probe_after_epoch(
                    model,
                    model_without_ddp,
                    optimizer,
                    args,
                    epoch,
                    train_steps,
                    logger=logger,
                    student_ema=student_ema,
                )
            except Exception:
                logger.exception(
                    "Linear probe eval failed after epoch %d; continuing training",
                    epoch + 1,
                )
                reset_denoiser_training_state(
                    model, model_without_ddp, optimizer, logger=logger
                )
            if misc.is_main_process():
                logger.info(
                    "Linear probe eval after epoch %d finished in %s",
                    epoch + 1,
                    str(datetime.timedelta(seconds=int(time.time() - linear_start))),
                )

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    logger.info("Training time: %s", str(datetime.timedelta(seconds=int(total_time))))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    apply_eval_defaults(args)
    apply_loss_weight_aliases(args)
    if args.ckpt_every_epoch is not None:
        args.save_freq = args.ckpt_every_epoch
    if args.results_dir is not None and not args.auto_experiment_dir:
        args.output_dir = args.results_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
