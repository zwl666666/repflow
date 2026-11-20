# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from copy import deepcopy
from datetime import timedelta
from glob import glob
from time import time
import argparse
import os
import pickle
import math
import json
import re
import shutil
import subprocess
import sys

import torch.nn as nn
from PIL import Image

from models import SiT_models, LatentProjector, LatentStudentNet, SimpleHead
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import (
    center_crop_arr,
    cleanup,
    create_logger,
    masked_smooth_l1_loss,
    parse_transport_args,
    requires_grad,
    sra_alignment_loss,
    sra_layer_index,
    token_level_cosine_loss,
    transport_training_losses_with_features,
    update_ema,
)
from repa_utils import (
    build_repa_mlp,
    dinov2_embed_dim,
    encoder_depth_to_layer_index,
    extract_dinov2_patch_tokens,
    load_dinov2_encoder,
    preprocess_for_dinov2,
    repa_neg_cosine_loss,
)
import wandb_utils
from s3_imagefolder import S3ImageFolder
from linear_student import run_linear_probe


#################################################################################
#                                  Training Loop                                #
#################################################################################

def setup_distributed_training(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    if args.bf16:
        assert hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported(), \
            "Current CUDA device does not support bf16 autocast."
    # NCCL's default collective timeout is 10 minutes. Periodic FID evaluation
    # runs the (TensorFlow) evaluator subprocess on rank 0 only, while ranks 1..N-1
    # idle at a dist.barrier(). Scoring 50k samples can exceed 10 minutes, which
    # trips the NCCL watchdog and aborts the whole process group (SIGABRT), killing
    # training exactly at the eval stage. Use a generous, configurable timeout so
    # rank-0-heavy eval no longer tears down the run.
    dist_timeout = timedelta(minutes=float(getattr(args, "dist_timeout_minutes", 180)))
    dist.init_process_group("nccl", timeout=dist_timeout)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    return rank, device


def get_effective_resume(args):
    return args.resume if args.resume is not None else args.ckpt


def student_ema_enabled(args):
    return args.student_ema or args.student_clean_ema


def sra_enabled(args):
    return float(getattr(args, "sra_loss_weight", 0.0) or 0.0) > 0


def drop_class_labels(labels, dropout_prob, num_classes):
    """Drop labels once so online and EMA SRA forwards share the same CFG mask."""
    if dropout_prob is None or dropout_prob <= 0:
        return labels
    drop = torch.rand(labels.shape[0], device=labels.device) < dropout_prob
    return torch.where(drop, torch.full_like(labels, num_classes), labels)


def _results_run_tag(results_dir: str) -> str:
    """Basename of --results-dir, e.g. results_sit_uncond_dual_wo_t."""
    return os.path.basename(os.path.normpath(results_dir))


def _checkpoint_dir_for_run(args, experiment_name: str) -> str:
    """Place checkpoints under ckpt_dir/<results_run_tag>/<experiment_name>/checkpoints."""
    run_tag = _results_run_tag(args.results_dir)
    return f"{args.ckpt_dir}/{run_tag}/{experiment_name}/checkpoints"


def _experiment_name_from_checkpoint_dir(checkpoint_dir):
    """checkpoint_dir is .../<experiment_name>/checkpoints (optionally under a run tag)."""
    return os.path.basename(os.path.dirname(checkpoint_dir.rstrip("/")))


def setup_experiment_directory(args, rank):
    resume_path = get_effective_resume(args)
    if rank == 0:
        if resume_path is not None:
            checkpoint_dir = os.path.dirname(resume_path)
            experiment_name = _experiment_name_from_checkpoint_dir(checkpoint_dir)
            if args.ckpt_dir:
                experiment_dir = f"{args.results_dir}/{experiment_name}"
            else:
                experiment_dir = os.path.dirname(checkpoint_dir)
        else:
            os.makedirs(args.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            experiment_name = (
                f"{experiment_index:03d}-{model_string_name}-"
                f"{args.path_type}-{args.prediction}-{args.loss_weight}"
            )
            experiment_dir = f"{args.results_dir}/{experiment_name}"
            if args.ckpt_dir:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                checkpoint_dir = _checkpoint_dir_for_run(args, experiment_name)
            else:
                checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        if resume_path is not None:
            logger.info(f"Resuming experiment at {experiment_dir}")
        else:
            logger.info(f"Experiment directory created at {experiment_dir}")
        if args.ckpt_dir:
            logger.info(
                f"Checkpoints directory: {checkpoint_dir} "
                f"(results run tag: {_results_run_tag(args.results_dir)})"
            )

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        checkpoint_dir = None
        experiment_dir = None
        logger = create_logger(None)
    shared = [checkpoint_dir, experiment_dir]
    dist.broadcast_object_list(shared, src=0)
    checkpoint_dir, experiment_dir = shared
    return logger, checkpoint_dir, experiment_dir


def get_label_dropout_prob(args):
    label_dropout_prob = args.class_dropout_prob
    if args.uncondition and label_dropout_prob <= 0:
        label_dropout_prob = 1e-8
    return label_dropout_prob


def get_student_input_config(args, teacher_patch_size):
    if args.student_direct_image_input:
        return args.image_size, teacher_patch_size * 8, 3
    return args.image_size // 8, teacher_patch_size, args.latent_channels


def student_feature_layers_list(args):
    """
    List of SiT block indices (after each block) used for teacher distillation features.
    Order is preserved; features are concatenated on the last dimension.
    """
    if args.student_feature_layers is not None and args.student_feature_layers.strip():
        return [int(x.strip()) for x in args.student_feature_layers.split(",") if x.strip()]
    return [args.student_feature_layer]


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


def projector_feature_layers_list(args):
    """
    Student block indices concatenated on the channel dim as projector input.
    Default is the last student layer only (backward compatible).
    """
    depth = int(args.student_depth)
    if args.projector_feature_layers is not None and str(args.projector_feature_layers).strip():
        layers = [int(x.strip()) for x in str(args.projector_feature_layers).split(",") if x.strip()]
        if not layers:
            return uniform_block_indices(depth, int(getattr(args, "projector_num_layers", 1) or 1))
        resolved = []
        for fl in layers:
            idx = fl if fl >= 0 else depth + fl
            if not (0 <= idx < depth):
                raise IndexError(
                    f"projector feature layer {fl} resolves to {idx}, out of range for student depth {depth}"
                )
            resolved.append(idx)
        last = depth - 1
        if last not in resolved:
            resolved.append(last)
        return resolved
    n = int(getattr(args, "projector_num_layers", 1) or 1)
    return uniform_block_indices(depth, n)


def build_models(args, device):
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    label_dropout_prob = get_label_dropout_prob(args)

    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=args.latent_channels,
        num_classes=args.num_classes,
        class_dropout_prob=label_dropout_prob,
    )
    sit_hidden = model.pos_embed.shape[-1]
    student = None
    student_ema = None
    projector = None
    align_head = None
    if args.student_loss_weight > 0:
        student_input_size, student_patch_size, student_in_channels = get_student_input_config(args, model.patch_size)
        student = LatentStudentNet(
            input_size=student_input_size,
            patch_size=student_patch_size,
            in_channels=student_in_channels,
            hidden_size=args.student_hidden_size,
            depth=args.student_depth,
            mlp_ratio=args.student_mlp_ratio,
            mask_ratio=args.student_mask_ratio,
        )
        if student_ema_enabled(args):
            student_ema = deepcopy(student).to(device)
            requires_grad(student_ema, False)
        feat_layers = student_feature_layers_list(args)
        proj_layers = projector_feature_layers_list(args)
        teacher_feat_dim = model.pos_embed.shape[-1] * len(feat_layers)
        projector = LatentProjector(
            input_size=student_input_size,
            patch_size=student_patch_size,
            student_hidden_size=args.student_hidden_size * len(proj_layers),
            teacher_hidden_size=teacher_feat_dim,
            projector_hidden_size=args.projector_hidden_size,
            depth=args.projector_depth,
            mlp_ratio=args.projector_mlp_ratio,
        )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(
        model.to(device),
        device_ids=[device],
        find_unused_parameters=bool(getattr(args, "freeze_generator", False)),
    )
    if student is not None:
        student = DDP(
            student.to(device),
            device_ids=[device],
            find_unused_parameters=bool(getattr(args, "freeze_student", False)),
        )
        if student_ema is not None:
            student_ema.eval()
        projector = DDP(projector.to(device), device_ids=[device])
        if args.student_align_cosine_weight > 0:
            # Project teacher tokens (per-layer dim sit_hidden) -> student token dim for cosine vs clean student.
            align_head = nn.Sequential(
                nn.Linear(sit_hidden, args.student_hidden_size),
                nn.LayerNorm(args.student_hidden_size),
                nn.Linear(args.student_hidden_size, args.student_hidden_size)
            ).to(device)
            align_head = DDP(align_head, device_ids=[device])

    # SRA (self-representation alignment): extra MLP only; not attached to SiT so default ckpts stay compatible.
    sra_head = None
    if sra_enabled(args):
        sra_head = SimpleHead(sit_hidden, sit_hidden).to(device)
        sra_head = DDP(sra_head, device_ids=[device])

    # REPA (DINOv2): default off when --repa-proj-coeff == 0
    repa_encoder = None
    repa_projector = None
    if float(getattr(args, "repa_proj_coeff", 0.0) or 0.0) > 0:
        z_dim = dinov2_embed_dim(args.repa_enc_type)
        repa_projector = build_repa_mlp(
            sit_hidden,
            int(args.repa_projector_dim),
            z_dim,
        ).to(device)
        repa_projector = DDP(repa_projector, device_ids=[device])
        repa_encoder = load_dinov2_encoder(
            args.repa_enc_type,
            device=device,
            resolution=int(args.image_size),
            repo_or_dir=getattr(args, "repa_encoder_repo", None),
            checkpoint_path=getattr(args, "repa_encoder_ckpt", None),
        )

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-ema").to(device)
    requires_grad(vae, False)
    vae.eval()
    return (
        model,
        student,
        student_ema,
        projector,
        align_head,
        ema,
        transport,
        transport_sampler,
        vae,
        latent_size,
        repa_encoder,
        repa_projector,
        sra_head,
    )


def build_optimizer(args, model, student, projector, align_head, repa_projector=None, sra_head=None):
    params = list(model.parameters())
    if student is not None:
        params += list(student.parameters())
    if projector is not None:
        params += list(projector.parameters())
    if align_head is not None:
        params += list(align_head.parameters())
    if repa_projector is not None:
        params += list(repa_projector.parameters())
    if sra_head is not None:
        params += list(sra_head.parameters())
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def build_dataloader(args, rank):
    world_size = dist.get_world_size()
    per_device_batch_size = int(args.global_batch_size // world_size)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    if args.data_path.startswith("s3://"):
        petrel_conf = getattr(args, "petrel_conf", None)
        if args.s3_index_cache:
            if rank == 0:
                if not os.path.isfile(args.s3_index_cache):
                    idx = S3ImageFolder.build_index(
                        args.data_path, petrel_conf=petrel_conf
                    )
                    d = os.path.dirname(args.s3_index_cache)
                    if d:
                        os.makedirs(d, exist_ok=True)
                    with open(args.s3_index_cache, "wb") as f:
                        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
            dist.barrier()
            with open(args.s3_index_cache, "rb") as f:
                idx = pickle.load(f)
            dataset = S3ImageFolder(
                args.data_path,
                transform=transform,
                index=idx,
                petrel_conf=petrel_conf,
            )
        else:
            dataset = S3ImageFolder(
                args.data_path, transform=transform, petrel_conf=petrel_conf
            )
    else:
        if not os.path.isdir(args.data_path):
            raise FileNotFoundError(f"Local --data-path not found: {args.data_path}")
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader, sampler, dataset


def log_training_metrics(logger, device, train_steps, log_steps, running_loss, running_sit_loss, running_student_loss, start_time, running_sra_loss=0.0, log_sra=False):
    torch.cuda.synchronize()
    end_time = time()
    steps_per_sec = log_steps / (end_time - start_time)
    avg_loss = torch.tensor(running_loss / log_steps, device=device)
    avg_sit_loss = torch.tensor(running_sit_loss / log_steps, device=device)
    avg_student_loss = torch.tensor(running_student_loss / log_steps, device=device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_sit_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_student_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    avg_sit_loss = avg_sit_loss.item() / dist.get_world_size()
    avg_student_loss = avg_student_loss.item() / dist.get_world_size()
    extra = ""
    if log_sra:
        avg_sra_loss = torch.tensor(running_sra_loss / log_steps, device=device)
        dist.all_reduce(avg_sra_loss, op=dist.ReduceOp.SUM)
        avg_sra_loss = avg_sra_loss.item() / dist.get_world_size()
        extra = f", SRA Loss: {avg_sra_loss:.4f}"
    logger.info(
        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
        f"SiT Loss: {avg_sit_loss:.4f}, Student Loss: {avg_student_loss:.4f}"
        f"{extra}, "
        f"Train Steps/Sec: {steps_per_sec:.2f}"
    )
    return end_time, avg_loss, steps_per_sec


def load_checkpoint(args, model, student, student_ema, projector, align_head, ema, opt, logger, repa_projector=None, sra_head=None):
    resume_path = get_effective_resume(args)
    if resume_path is None:
        return {"start_epoch": 0, "resume_step_in_epoch": 0, "train_steps": 0}

    assert os.path.isfile(resume_path), f"Resume checkpoint not found: {resume_path}"
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.module.load_state_dict(checkpoint["model"])
    if student is not None:
        if "student" not in checkpoint or "projector" not in checkpoint:
            raise RuntimeError("Checkpoint is missing student/projector weights; cannot resume with student_loss_weight > 0.")
        student.module.load_state_dict(checkpoint["student"])
        projector.module.load_state_dict(checkpoint["projector"])
        if align_head is not None and checkpoint.get("align_head") is not None:
            align_head.module.load_state_dict(checkpoint["align_head"])
        if student_ema is not None:
            if checkpoint.get("student_ema") is not None:
                student_ema.load_state_dict(checkpoint["student_ema"])
            else:
                update_ema(student_ema, student.module, decay=0)
                logger.warning("Checkpoint missing student_ema; initialized from current student weights.")
    elif checkpoint.get("student") is not None or checkpoint.get("projector") is not None:
        logger.warning(
            "Checkpoint contains student/projector weights but student_loss_weight==0 (baseline); "
            "those weights are ignored."
        )
    if repa_projector is not None:
        if checkpoint.get("repa_projector") is not None:
            repa_projector.module.load_state_dict(checkpoint["repa_projector"])
        else:
            logger.warning("Checkpoint missing repa_projector; continuing with freshly initialized REPA projector.")
    if sra_head is not None:
        if checkpoint.get("sra_head") is not None:
            sra_head.module.load_state_dict(checkpoint["sra_head"])
        else:
            logger.warning("Checkpoint missing sra_head; continuing with a freshly initialized SRA head.")
    ema.load_state_dict(checkpoint["ema"])
    try:
        opt.load_state_dict(checkpoint["opt"])
    except Exception as e:
        logger.warning(f"Could not load optimizer state ({e}); continuing with a freshly initialized optimizer state.")
    resume_state = {
        "start_epoch": checkpoint.get("epoch", 0),
        "resume_step_in_epoch": checkpoint.get("step_in_epoch", 0),
        "train_steps": checkpoint.get("train_steps", 0),
    }
    logger.info(
        f"Resumed checkpoint from {resume_path} "
        f"(epoch={resume_state['start_epoch']}, step_in_epoch={resume_state['resume_step_in_epoch']}, "
        f"train_steps={resume_state['train_steps']})"
    )
    return resume_state


def move_optimizer_to_device(opt, device):
    for opt_state in opt.state.values():
        for key, value in opt_state.items():
            if isinstance(value, torch.Tensor):
                opt_state[key] = value.to(device)


def _checkpoint_payload(epoch, step_in_epoch, train_steps, model, student, student_ema, projector, align_head, ema, opt, args, repa_projector=None, sra_head=None):
    payload = {
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "train_steps": train_steps,
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
    }
    if student is not None:
        payload["student"] = student.module.state_dict()
    if student_ema is not None:
        payload["student_ema"] = student_ema.state_dict()
    if projector is not None:
        payload["projector"] = projector.module.state_dict()
    if align_head is not None:
        payload["align_head"] = align_head.module.state_dict()
    if repa_projector is not None:
        payload["repa_projector"] = repa_projector.module.state_dict()
    if sra_head is not None:
        payload["sra_head"] = sra_head.module.state_dict()
    return payload


def should_trigger_epoch_interval(args, epoch):
    return args.ckpt_every_epoch > 0 and (epoch + 1) % args.ckpt_every_epoch == 0


def should_trigger_linear_eval(args, epoch):
    start_epoch = getattr(args, "linear_eval_start_epoch", 1)
    if epoch + 1 < start_epoch:
        return False
    return should_trigger_epoch_interval(args, epoch)


def _rclone_remote_checkpoint_path(args, local_checkpoint_path):
    remote_root = (args.ckpt_rclone_remote or "").strip().rstrip("/")
    if not remote_root:
        return None
    if args.ckpt_dir:
        rel = os.path.relpath(local_checkpoint_path, args.ckpt_dir)
    else:
        rel = os.path.basename(local_checkpoint_path)
    return f"{remote_root}/{rel}".replace("\\", "/")


def sync_checkpoint_to_rclone(local_checkpoint_path, args, logger):
    remote_path = _rclone_remote_checkpoint_path(args, local_checkpoint_path)
    if remote_path is None:
        return
    try:
        proc = subprocess.run(
            ["rclone", "copyto", local_checkpoint_path, remote_path, "--retries", "3"],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if proc.returncode == 0:
            logger.info(f"Synced checkpoint to rclone: {remote_path}")
        else:
            stderr = (proc.stderr or proc.stdout or "").strip()
            logger.warning(f"rclone sync failed (exit {proc.returncode}): {stderr}")
    except subprocess.TimeoutExpired:
        logger.warning("rclone checkpoint sync timed out after 3600s")
    except FileNotFoundError:
        logger.warning("rclone not found in PATH; skipped remote checkpoint sync")
    except Exception as exc:
        logger.warning(f"rclone checkpoint sync error: {exc}")


def save_checkpoint(rank, checkpoint_dir, epoch, step_in_epoch, train_steps, model, student, student_ema, projector, align_head, ema, opt, args, logger, repa_projector=None, sra_head=None):
    if train_steps % args.ckpt_every != 0 or train_steps == 0:
        return
    if rank == 0:
        checkpoint = _checkpoint_payload(
            epoch, step_in_epoch, train_steps, model, student, student_ema, projector, align_head, ema, opt, args, repa_projector=repa_projector, sra_head=sra_head
        )
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        sync_checkpoint_to_rclone(checkpoint_path, args, logger)
    dist.barrier()


def save_epoch_checkpoint(rank, checkpoint_dir, epoch, train_steps, model, student, student_ema, projector, align_head, ema, opt, args, logger, repa_projector=None, sra_head=None):
    if not should_trigger_epoch_interval(args, epoch):
        return
    if rank == 0:
        checkpoint = _checkpoint_payload(
            epoch + 1, 0, train_steps, model, student, student_ema, projector, align_head, ema, opt, args, repa_projector=repa_projector, sra_head=sra_head
        )
        checkpoint_path = f"{checkpoint_dir}/latest.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved latest checkpoint to {checkpoint_path} (epoch={epoch + 1})")
        sync_checkpoint_to_rclone(checkpoint_path, args, logger)
    dist.barrier()


def _build_eval_sampling_inputs(args, batch_size, latent_size, device):
    use_cfg = (args.eval_cfg_scale > 1.0) and (not args.uncondition)
    if args.uncondition:
        y = torch.full((batch_size,), args.num_classes, device=device, dtype=torch.long)
    else:
        y = torch.randint(0, args.num_classes, (batch_size,), device=device)
    z = torch.randn(batch_size, args.latent_channels, latent_size, latent_size, device=device)
    if use_cfg:
        z = torch.cat([z, z], dim=0)
        y_null = torch.full((batch_size,), args.num_classes, device=device, dtype=torch.long)
        y = torch.cat([y, y_null], dim=0)
    return z, y, use_cfg


def _sample_for_fid(args, state, sample_dir, epoch):
    rank = state["rank"]
    logger = state["logger"]
    device = state["device"]
    ema = state["ema"]
    transport_sampler = state["transport_sampler"]
    vae = state["vae"]
    latent_size = state["latent_size"]

    n = args.eval_per_proc_batch_size
    world_size = dist.get_world_size()
    global_batch_size = n * world_size
    total_samples = int(math.ceil(args.eval_num_fid_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = total_samples // world_size
    iterations = samples_needed_this_gpu // n

    if rank == 0:
        os.makedirs(sample_dir, exist_ok=True)
        logger.info(
            f"FID eval epoch={epoch + 1}: sampling {total_samples} images "
            f"(target={args.eval_num_fid_samples}) into {sample_dir}"
        )
    dist.barrier()

    sample_fn = transport_sampler.sample_ode(
        sampling_method=args.eval_sampling_method,
        num_steps=args.eval_num_sampling_steps,
        atol=args.eval_atol,
        rtol=args.eval_rtol,
    )

    for it in range(iterations):
        z, y, use_cfg = _build_eval_sampling_inputs(args, n, latent_size, device)
        if use_cfg:
            model_kwargs = dict(y=y, cfg_scale=args.eval_cfg_scale)
            model_fn = ema.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fn = ema.forward

        with torch.no_grad():
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if use_cfg:
                samples, _ = samples.chunk(2, dim=0)
            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        base = it * global_batch_size
        for idx, sample in enumerate(samples):
            global_index = base + idx * world_size + rank
            Image.fromarray(sample).save(f"{sample_dir}/{global_index:06d}.png")
        if (it + 1) % max(1, args.eval_log_interval) == 0 and rank == 0:
            logger.info(f"FID eval sampling progress: {it + 1}/{iterations} iterations")
        dist.barrier()


def _create_npz_from_sample_folder(sample_dir, num, logger):
    samples = []
    for i in range(num):
        sample_path = f"{sample_dir}/{i:06d}.png"
        sample_pil = Image.open(sample_path)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    logger.info(f"Saved sample npz to {npz_path} [shape={samples.shape}]")
    return npz_path


def _compute_metrics_with_evaluator(args, sample_npz_path, logger):
    if not args.fid_ref_path:
        logger.warning("FID evaluation skipped: --fid-ref-path not set.")
        return {}
    if not args.evaluator_script:
        logger.warning("FID evaluation skipped: --evaluator-script not set.")
        return {}

    cmd = [
        sys.executable,
        args.evaluator_script,
        args.fid_ref_path,
        sample_npz_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = proc.stdout.strip()
    if proc.returncode != 0:
        logger.error(f"Evaluator command failed ({proc.returncode}). Output:\n{output}")
        return {}
    logger.info(f"evaluator.py output:\n{output}")

    metrics = {}
    patterns = {
        "inception_score": r"Inception Score:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        "fid": r"FID:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        "sfid": r"sFID:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        "precision": r"Precision:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        "recall": r"Recall:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, output)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def _append_eval_metrics_record(metrics_all_path, record, logger):
    """
    Append/update one epoch record into a single metrics file for the whole run.
    """
    history = []
    if os.path.isfile(metrics_all_path):
        try:
            with open(metrics_all_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                history = loaded
        except Exception as e:
            logger.warning(f"Failed to read existing eval summary ({e}); recreating {metrics_all_path}.")

    # If an epoch record already exists (e.g. rerun), replace it.
    history = [x for x in history if not (isinstance(x, dict) and x.get("epoch") == record.get("epoch"))]
    history.append(record)
    history.sort(key=lambda x: x.get("epoch", 0))

    tmp_path = metrics_all_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, metrics_all_path)
    logger.info(f"Updated eval summary at {metrics_all_path}")


def _merge_epoch_eval_metrics(experiment_dir, epoch, extra_fields, logger):
    eval_root = f"{experiment_dir}/eval"
    stage_name = f"epoch_{epoch + 1:04d}"
    metric_path = f"{eval_root}/{stage_name}/metrics.json"
    metrics_all_path = f"{eval_root}/metrics_all.json"

    record = {"epoch": epoch + 1}
    if os.path.isfile(metric_path):
        try:
            with open(metric_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                record.update(loaded)
        except Exception as e:
            logger.warning(f"Failed to read existing metrics at {metric_path} ({e}); creating a new record.")

    record.update(extra_fields)
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved eval metrics to {metric_path}")
    _append_eval_metrics_record(metrics_all_path, record, logger)
    return record


def _build_linear_probe_args(args, state):
    world_size = dist.get_world_size()
    per_gpu_batch_size = max(1, args.linear_global_batch_size // world_size)
    return argparse.Namespace(
        model=args.model,
        image_size=args.image_size,
        num_classes=args.num_classes,
        latent_channels=args.latent_channels,
        vae=args.vae,
        vae_model=None,
        student_depth=args.student_depth,
        student_hidden_size=args.student_hidden_size,
        student_mlp_ratio=args.student_mlp_ratio,
        student_mask_ratio=args.student_mask_ratio,
        student_direct_image_input=args.student_direct_image_input,
        train_data_path=args.linear_train_data_path,
        val_data_path=args.linear_val_data_path,
        train_label_file=args.linear_train_label_file,
        val_label_file=args.linear_val_label_file,
        s3_endpoint=args.s3_endpoint,
        s3_region=args.s3_region,
        petrel_conf=getattr(args, "petrel_conf", None),
        epochs=args.linear_epochs,
        batch_size=per_gpu_batch_size,
        lr=args.linear_lr,
        num_workers=args.linear_num_workers,
        use_amp=args.linear_use_amp,
        pin_mem=True,
        ckpt=None,
        resume=None,
    )


def _get_epoch_eval_record(experiment_dir, display_epoch):
    metrics_all_path = f"{experiment_dir}/eval/metrics_all.json"
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


def maybe_run_pending_linear_eval_on_resume(args, state, resume_state):
    """
    When resuming from a checkpoint saved at an eval interval (e.g. epoch 100),
    run linear probe first if that epoch's metrics are missing linear_best_acc.
    """
    if get_effective_resume(args) is None:
        return
    if not args.linear_eval_during_train or not args.uncondition:
        return
    if args.student_loss_weight <= 0 or state["student"] is None:
        return
    if resume_state["resume_step_in_epoch"] != 0:
        return

    display_epoch = resume_state["start_epoch"]
    if display_epoch <= 0:
        return

    completed_epoch_index = display_epoch - 1
    if display_epoch < getattr(args, "linear_eval_start_epoch", 1):
        return
    if not should_trigger_epoch_interval(args, completed_epoch_index):
        return

    record = _get_epoch_eval_record(state["experiment_dir"], display_epoch)
    if record is not None and "linear_best_acc" in record:
        return

    rank = state["rank"]
    logger = state["logger"]
    if rank == 0:
        logger.info(
            f"Resume at epoch {display_epoch}: running pending linear probe eval before continuing training"
        )
    evaluate_linear_probe_after_epoch(args, state, completed_epoch_index, force=True)


def evaluate_linear_probe_after_epoch(args, state, epoch, force=False):
    if not args.linear_eval_during_train:
        return
    if not args.uncondition:
        return
    if args.student_loss_weight <= 0 or state["student"] is None:
        return
    if not force and not should_trigger_linear_eval(args, epoch):
        return

    rank = state["rank"]
    logger = state["logger"]
    if not args.linear_train_data_path or not args.linear_val_data_path:
        if rank == 0:
            logger.warning("Skip linear probe eval because --linear-train-data-path / --linear-val-data-path is not set.")
        return

    train_steps = state["train_steps"]
    experiment_dir = state["experiment_dir"]
    student_ema = state.get("student_ema")
    student = student_ema if student_ema is not None else state["student"]
    student_was_training = student.training
    weight_source = "student_ema" if student_ema is not None else "student"

    if rank == 0:
        logger.info(
            f"Linear probe eval epoch={epoch + 1}: training linear head on frozen {weight_source} features"
        )
    student.eval()
    linear_args = _build_linear_probe_args(args, state)
    result = run_linear_probe(
        linear_args,
        device=state["device"],
        local_rank=state["rank"],
        student=student,
    )
    if student_was_training:
        student.train()
    dist.barrier()

    if rank == 0:
        metrics = {
            "epoch": epoch + 1,
            "train_steps": train_steps,
            "linear_best_acc": result["best_acc"],
            "linear_final_acc": result["final_acc"],
            "linear_epochs": args.linear_epochs,
            "linear_global_batch_size": args.linear_global_batch_size,
            "linear_train_data_path": args.linear_train_data_path,
            "linear_val_data_path": args.linear_val_data_path,
            "linear_train_label_file": args.linear_train_label_file,
            "linear_val_label_file": args.linear_val_label_file,
        }
        _merge_epoch_eval_metrics(experiment_dir, epoch, metrics, logger)
        if args.wandb:
            wandb_utils.log({"linear_best_acc": result["best_acc"]}, step=train_steps)
    dist.barrier()


def evaluate_fid_after_epoch(args, state, epoch):
    if not args.eval_during_train:
        return
    if not should_trigger_epoch_interval(args, epoch):
        return

    rank = state["rank"]
    logger = state["logger"]
    if not args.fid_ref_path:
        if rank == 0:
            logger.warning("Skip periodic FID eval because --fid-ref-path is not set.")
        return
    if not args.evaluator_script:
        if rank == 0:
            logger.warning("Skip periodic FID eval because --evaluator-script is not set.")
        return

    train_steps = state["train_steps"]
    experiment_dir = state["experiment_dir"]

    stage_name = f"epoch_{epoch + 1:04d}"
    eval_root = f"{experiment_dir}/eval"
    stage_dir = f"{eval_root}/{stage_name}"
    if args.eval_temp_root:
        experiment_name = os.path.basename(experiment_dir.rstrip("/"))
        run_tag = _results_run_tag(args.results_dir)
        sample_dir = f"{args.eval_temp_root}/{run_tag}/{experiment_name}/{stage_name}/samples"
    else:
        sample_dir = f"{stage_dir}/samples"
    metric_path = f"{stage_dir}/metrics.json"
    metrics_all_path = f"{eval_root}/metrics_all.json"

    if rank == 0:
        os.makedirs(eval_root, exist_ok=True)
        if os.path.isdir(stage_dir):
            shutil.rmtree(stage_dir)
        os.makedirs(stage_dir, exist_ok=True)
        if os.path.isdir(sample_dir):
            shutil.rmtree(sample_dir)
    dist.barrier()

    _sample_for_fid(args, state, sample_dir, epoch)
    dist.barrier()

    if rank == 0:
        sample_npz_path = _create_npz_from_sample_folder(sample_dir, args.eval_num_fid_samples, logger)
        eval_metrics = _compute_metrics_with_evaluator(args, sample_npz_path, logger)
        remove_temporary_files = not args.keep_eval_samples
        metrics = {
            "epoch": epoch + 1,
            "train_steps": train_steps,
            "num_fid_samples": args.eval_num_fid_samples,
            "fid_ref_path": args.fid_ref_path,
            "sample_npz_path": None if remove_temporary_files else sample_npz_path,
            "temporary_files_removed": remove_temporary_files,
            **eval_metrics,
        }
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved eval metrics to {metric_path}")
        _append_eval_metrics_record(metrics_all_path, metrics, logger)
        if args.wandb and "fid" in eval_metrics:
            wandb_utils.log({"fid": eval_metrics["fid"]}, step=train_steps)
        if remove_temporary_files:
            shutil.rmtree(sample_dir, ignore_errors=True)
            logger.info(f"Removed temporary eval samples at {sample_dir}")
            if os.path.isfile(sample_npz_path):
                os.remove(sample_npz_path)
                logger.info(f"Removed temporary eval npz at {sample_npz_path}")
    dist.barrier()


def build_sampling_inputs(args, local_batch_size, latent_size, device, ema):
    use_cfg = (args.cfg_scale > 1.0) and (not args.uncondition)
    if args.uncondition:
        ys = torch.full((local_batch_size,), args.num_classes, device=device, dtype=torch.long)
    else:
        ys = torch.randint(args.num_classes, size=(local_batch_size,), device=device)
    zs = torch.randn(local_batch_size, args.latent_channels, latent_size, latent_size, device=device)

    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.full((local_batch_size,), args.num_classes, device=device, dtype=torch.long)
        ys = torch.cat([ys, y_null], 0)
        return zs, ema.forward_with_cfg, dict(y=ys, cfg_scale=args.cfg_scale), True
    return zs, ema.forward, dict(y=ys), False


def maybe_log_samples(args, state):
    train_steps = state["train_steps"]
    if train_steps % args.sample_every != 0 or train_steps == 0:
        return

    logger = state["logger"]
    device = state["device"]
    transport_sampler = state["transport_sampler"]
    ema = state["ema"]
    vae = state["vae"]
    latent_size = state["latent_size"]
    local_batch_size = state["local_batch_size"]
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    logger.info("Generating EMA samples...")
    zs, model_fn, sample_model_kwargs, use_cfg = build_sampling_inputs(args, local_batch_size, latent_size, device, ema)
    with torch.no_grad():
        sample_fn = transport_sampler.sample_ode()
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=args.bf16):
            samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
            if use_cfg:
                samples, _ = samples.chunk(2, dim=0)
            samples = vae.decode(samples / 0.18215).sample
        out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
        dist.all_gather_into_tensor(out_samples, samples)
    dist.barrier()

    if args.wandb:
        wandb_utils.log_image(out_samples, train_steps)
    logger.info("Generating EMA samples done.")


def train_one_epoch(args, epoch, loader, sampler, state):
    logger = state["logger"]
    device = state["device"]
    rank = state["rank"]
    checkpoint_dir = state["checkpoint_dir"]
    model = state["model"]
    student = state["student"]
    projector = state["projector"]
    align_head = state["align_head"]
    student_ema = state["student_ema"]
    ema = state["ema"]
    transport = state["transport"]
    vae = state["vae"]
    opt = state["opt"]
    repa_encoder = state["repa_encoder"]
    repa_projector = state["repa_projector"]
    sra_head = state["sra_head"]

    sampler.set_epoch(epoch)
    logger.info(f"Beginning epoch {epoch}...")

    train_steps = state["train_steps"]
    log_steps = state["log_steps"]
    running_loss = state["running_loss"]
    running_sit_loss = state["running_sit_loss"]
    running_student_loss = state["running_student_loss"]
    running_sra_loss = state["running_sra_loss"]
    start_time = state["start_time"]
    resume_step_in_epoch = state["resume_step_in_epoch"] if epoch == state["start_epoch"] else 0
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32
    repa_coeff = float(getattr(args, "repa_proj_coeff", 0.0) or 0.0)
    sra_coeff = float(getattr(args, "sra_loss_weight", 0.0) or 0.0)
    use_sra = sra_coeff > 0 and sra_head is not None
    sra_s_layer = sra_layer_index(args.sra_block_out_s) if use_sra else None
    sra_t_layer = sra_layer_index(args.sra_block_out_t) if use_sra else None
    repa_layer_kw = (
        encoder_depth_to_layer_index(int(args.repa_encoder_depth))
        if repa_coeff > 0 and repa_projector is not None
        else None
    )

    for step_in_epoch, (x, y) in enumerate(loader):
        if step_in_epoch < resume_step_in_epoch:
            continue
        if args.max_steps_per_epoch > 0 and step_in_epoch >= resume_step_in_epoch + args.max_steps_per_epoch:
            break
        x = x.to(device)
        y = y.to(device)
        train_y = torch.full_like(y, args.num_classes) if args.uncondition else y
        if use_sra and not args.uncondition:
            train_y = drop_class_labels(train_y, args.class_dropout_prob, args.num_classes)
        image_x = x
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=args.bf16):
                latent_x = vae.encode(image_x).latent_dist.sample().mul_(0.18215)

        align_layer_kw = None
        if (
            align_head is not None
            and args.student_align_cosine_weight > 0
            and student is not None
            and args.student_loss_weight > 0
        ):
            align_layer_kw = args.student_align_teacher_layer
            if align_layer_kw is None:
                align_layer_kw = state["student_feature_layers"][0]
        elif use_sra:
            align_layer_kw = sra_s_layer

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=args.bf16):
            loss_dict, teacher_feat, t, teacher_align_feat, teacher_repa_feat = transport_training_losses_with_features(
                transport,
                model,
                latent_x,
                dict(y=train_y),
                feature_layers=state["student_feature_layers"],
                return_teacher_features=student is not None,
                align_feature_layer=align_layer_kw,
                repa_feature_layer=repa_layer_kw,
            )
            sit_loss = loss_dict["loss"].mean()

            if student is not None and args.student_loss_weight > 0:
                student_x = image_x if args.student_direct_image_input else latent_x
                proj_layers = state["projector_feature_layers"]
                student_kw = {}
                if len(proj_layers) > 1:
                    student_kw["feature_layers"] = proj_layers
                student_tokens, student_mask, keep_idx = student(
                    student_x, use_mask=args.student_use_mask, return_mask=True, **student_kw
                )
                student_feat = projector(student_tokens, t, keep_idx if args.student_use_mask else None)
                if args.student_use_mask:
                    student_loss = masked_smooth_l1_loss(student_feat, teacher_feat.detach(), student_mask)
                else:
                    student_loss = masked_smooth_l1_loss(student_feat, teacher_feat.detach())
            else:
                student_loss = torch.zeros((), device=device)

            align_loss = torch.zeros((), device=device)
            if (
                align_head is not None
                and args.student_align_cosine_weight > 0
                and student is not None
                and args.student_loss_weight > 0
            ):
                with torch.no_grad():
                    clean_student = student_ema if student_ema is not None else student
                    student_clean = clean_student(student_x, use_mask=False, return_mask=False)
                student_clean_det = student_clean.detach()
                if teacher_align_feat is None:
                    raise RuntimeError("Cosine align requires teacher_align_feat from the teacher forward.")
                teacher_to_student = align_head(teacher_align_feat)
                align_loss = token_level_cosine_loss(student_clean_det, teacher_to_student)

            repa_loss = torch.zeros((), device=device)
            if repa_coeff > 0 and repa_projector is not None:
                if teacher_repa_feat is None:
                    raise RuntimeError("REPA enabled but teacher_repa_feat is None.")
                if repa_encoder is None:
                    raise RuntimeError("REPA enabled but DINOv2 encoder is not loaded.")
                N, T, D = teacher_repa_feat.shape
                z_tilde = repa_projector(teacher_repa_feat.reshape(N * T, D)).reshape(N, T, -1)
                with torch.no_grad():
                    dino_in = preprocess_for_dinov2(
                        image_x, resolution=args.image_size, input_range="neg1_1"
                    )
                    z_dino = extract_dinov2_patch_tokens(repa_encoder, dino_in)
                repa_loss = repa_neg_cosine_loss(z_dino, z_tilde)

            sra_loss = torch.zeros((), device=device)
            if use_sra:
                if teacher_align_feat is None:
                    raise RuntimeError("SRA enabled but online layer features are None.")
                x0, x1 = loss_dict["x0"], loss_dict["x1"]
                # SiT Linear: t=0 noise, t=1 data. Teacher is cleaner (larger t), opposite of SRA-main's t=0 data.
                t_teacher = (t + float(args.sra_t_max) * torch.rand_like(t)).clamp(0, 1)
                _, xt_teacher, _ = transport.path_sampler.plan(t_teacher, x0, x1)
                with torch.no_grad():
                    _, _, ema_sra_feat, _ = ema(
                        xt_teacher,
                        t_teacher,
                        train_y,
                        return_features=True,
                        feature_layers=[],
                        align_feature_layer=sra_t_layer,
                        use_dropout=False,
                    )
                if ema_sra_feat is None:
                    raise RuntimeError("SRA enabled but EMA later-layer features are None.")
                xr = sra_head(teacher_align_feat)
                sra_loss = sra_alignment_loss(xr, ema_sra_feat, args.sra_loss_type)

            loss = (
                sit_loss * (0.0 if getattr(args, "freeze_generator", False) else 1.0)
                + args.student_loss_weight * student_loss
                + args.student_align_cosine_weight * align_loss
                + repa_coeff * repa_loss
                + sra_coeff * sra_loss
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not getattr(args, "freeze_generator", False):
            update_ema(ema, model.module)
        if student_ema is not None and not getattr(args, "freeze_student", False):
            update_ema(student_ema, student.module)

        running_loss += loss.item()
        running_sit_loss += sit_loss.item()
        running_student_loss += student_loss.item()
        running_sra_loss += sra_loss.item()
        log_steps += 1
        train_steps += 1

        if train_steps % args.log_every == 0:
            start_time, avg_loss, steps_per_sec = log_training_metrics(
                logger,
                device,
                train_steps,
                log_steps,
                running_loss,
                running_sit_loss,
                running_student_loss,
                start_time,
                running_sra_loss=running_sra_loss,
                log_sra=use_sra,
            )
            if args.wandb:
                wandb_utils.log(
                    {
                        "train loss": avg_loss,
                        "train steps/sec": steps_per_sec,
                    },
                    step=train_steps,
                )
            running_loss = 0
            running_sit_loss = 0
            running_student_loss = 0
            running_sra_loss = 0
            log_steps = 0

        state["train_steps"] = train_steps
        # save_checkpoint(
        #     rank,
        #     checkpoint_dir,
        #     epoch,
        #     step_in_epoch + 1,
        #     train_steps,
        #     model,
        #     student,
        #     projector,
        #     align_head,
        #     ema,
        #     opt,
        #     args,
        #     logger,
        # )
        # maybe_log_samples(args, state)

    state.update(
        start_epoch=epoch + 1,
        resume_step_in_epoch=0,
        train_steps=train_steps,
        log_steps=log_steps,
        running_loss=running_loss,
        running_sit_loss=running_sit_loss,
        running_student_loss=running_student_loss,
        running_sra_loss=running_sra_loss,
        start_time=start_time,
    )
    save_epoch_checkpoint(
        rank,
        checkpoint_dir,
        epoch,
        train_steps,
        model,
        student,
        state["student_ema"],
        projector,
        align_head,
        ema,
        opt,
        args,
        logger,
        repa_projector=state["repa_projector"],
        sra_head=sra_head,
    )
    evaluate_fid_after_epoch(args, state, epoch)
    evaluate_linear_probe_after_epoch(args, state, epoch)


def configure_generator_train_mode(args, model, logger):
    if not getattr(args, "freeze_generator", False):
        model.train()
        return
    if get_effective_resume(args) is None:
        raise ValueError("--freeze-generator requires --resume from the joint-training checkpoint.")
    requires_grad(model.module, False)
    model.eval()
    frozen = sum(p.numel() for p in model.module.parameters())
    logger.info(
        f"Generator frozen ({frozen:,} parameters); only student/projector/alignment heads update. "
        "Generator EMA updates are disabled."
    )


def configure_student_train_mode(args, student, logger):
    if student is None:
        if getattr(args, "freeze_student", False):
            raise ValueError("--freeze-student requires student distillation (--student-loss-weight > 0).")
        return
    if not getattr(args, "freeze_student", False):
        student.train()
        return
    if get_effective_resume(args) is None:
        raise ValueError("--freeze-student requires --resume from the joint-training checkpoint.")
    requires_grad(student.module, False)
    student.eval()
    frozen = sum(p.numel() for p in student.module.parameters())
    logger.info(
        f"Student encoder frozen ({frozen:,} parameters); generator/projector/alignment heads continue. "
        "Student EMA updates are disabled; frozen features remain available for evaluation."
    )


def main(args):
    """
    Trains a new SiT model.
    """
    if args.student_align_cosine_weight > 0 and args.student_loss_weight <= 0:
        raise ValueError("--student-align-cosine-weight > 0 requires --student-loss-weight > 0 (student must exist).")
    if args.student_align_cosine_weight < 0:
        raise ValueError("--student-align-cosine-weight must be non-negative.")
    if student_ema_enabled(args) and args.student_loss_weight <= 0:
        raise ValueError("--student-ema requires --student-loss-weight > 0 (student must exist).")
    if int(getattr(args, "projector_num_layers", 1) or 1) < 1:
        raise ValueError("--projector-num-layers must be >= 1.")
    if float(getattr(args, "repa_proj_coeff", 0.0) or 0.0) < 0:
        raise ValueError("--repa-proj-coeff must be non-negative.")
    if getattr(args, "freeze_generator", False) and getattr(args, "freeze_student", False):
        raise ValueError("--freeze-generator and --freeze-student are mutually exclusive.")
    if sra_enabled(args):
        if args.student_loss_weight > 0:
            raise ValueError("SRA is teacher-only: use --student-loss-weight 0 (do not build a student).")
        if float(args.sra_loss_weight) < 0:
            raise ValueError("--sra-loss-weight must be non-negative.")
        if int(args.sra_block_out_s) >= int(args.sra_block_out_t):
            raise ValueError("--sra-block-out-s must be < --sra-block-out-t (1-based earlier vs later block).")

    rank, device = setup_distributed_training(args)
    logger, checkpoint_dir, experiment_dir = setup_experiment_directory(args, rank)

    (
        model,
        student,
        student_ema,
        projector,
        align_head,
        ema,
        transport,
        transport_sampler,
        vae,
        latent_size,
        repa_encoder,
        repa_projector,
        sra_head,
    ) = build_models(args, device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if student is not None:
        logger.info(f"Student Parameters: {sum(p.numel() for p in student.parameters()):,}")
        if student_ema is not None:
            ema_purposes = ["linear probe eval"]
            if args.student_align_cosine_weight > 0:
                ema_purposes.append("cosine align clean forward")
            logger.info(f"Student EMA enabled for: {', '.join(ema_purposes)}.")
        proj_layers = projector_feature_layers_list(args)
        logger.info(
            f"Projector student layers (concat dim=-1): {proj_layers} "
            f"(input dim={args.student_hidden_size * len(proj_layers)})"
        )
        logger.info(f"Projector Parameters: {sum(p.numel() for p in projector.parameters()):,}")
        if align_head is not None:
            logger.info(f"Align-head Parameters: {sum(p.numel() for p in align_head.parameters()):,}")
    else:
        logger.info("Student/projector not built (baseline: --student-loss-weight 0).")
    if repa_projector is not None:
        logger.info(
            f"REPA projector Parameters: {sum(p.numel() for p in repa_projector.parameters()):,} "
            f"(enc={args.repa_enc_type}, depth={args.repa_encoder_depth}, coeff={args.repa_proj_coeff})"
        )
    else:
        logger.info("REPA disabled (--repa-proj-coeff 0).")
    if sra_head is not None:
        logger.info(
            f"SRA enabled (no student): loss_weight={args.sra_loss_weight}, "
            f"block_out_s={args.sra_block_out_s} (0-based {sra_layer_index(args.sra_block_out_s)}), "
            f"block_out_t={args.sra_block_out_t} (0-based {sra_layer_index(args.sra_block_out_t)}), "
            f"t_max={args.sra_t_max}, loss_type={args.sra_loss_type}. "
            f"SRA-head Parameters: {sum(p.numel() for p in sra_head.parameters()):,}"
        )
    else:
        logger.info("SRA disabled (--sra-loss-weight 0).")
    logger.info(f"Autocast bf16: {'enabled' if args.bf16 else 'disabled'}")
    opt = build_optimizer(
        args, model, student, projector, align_head, repa_projector=repa_projector, sra_head=sra_head
    )
    loader, sampler, dataset = build_dataloader(args, rank)
    resume_state = load_checkpoint(
        args,
        model,
        student,
        student_ema,
        projector,
        align_head,
        ema,
        opt,
        logger,
        repa_projector=repa_projector,
        sra_head=sra_head,
    )
    if get_effective_resume(args) is not None:
        move_optimizer_to_device(opt, device)
    if resume_state["resume_step_in_epoch"] >= len(loader):
        resume_state["start_epoch"] += resume_state["resume_step_in_epoch"] // len(loader)
        resume_state["resume_step_in_epoch"] = resume_state["resume_step_in_epoch"] % len(loader)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    if get_effective_resume(args) is None:
        update_ema(ema, model.module, decay=0)
        if student_ema is not None:
            update_ema(student_ema, student.module, decay=0)
    configure_generator_train_mode(args, model, logger)
    if student is not None:
        configure_student_train_mode(args, student, logger)
        if student_ema is not None:
            student_ema.eval()
        projector.train()
    if align_head is not None:
        align_head.train()
    if repa_projector is not None:
        repa_projector.train()
    if sra_head is not None:
        sra_head.train()
    ema.eval()

    state = {
        "logger": logger,
        "rank": rank,
        "device": device,
        "checkpoint_dir": checkpoint_dir,
        "experiment_dir": experiment_dir,
        "model": model,
        "student": student,
        "student_ema": student_ema,
        "projector": projector,
        "align_head": align_head,
        "repa_encoder": repa_encoder,
        "repa_projector": repa_projector,
        "sra_head": sra_head,
        "ema": ema,
        "transport": transport,
        "transport_sampler": transport_sampler,
        "vae": vae,
        "opt": opt,
        "latent_size": latent_size,
        "student_feature_layers": student_feature_layers_list(args) if student is not None else [],
        "projector_feature_layers": projector_feature_layers_list(args) if student is not None else [],
        "local_batch_size": loader.batch_size,
        "start_epoch": resume_state["start_epoch"],
        "resume_step_in_epoch": resume_state["resume_step_in_epoch"],
        "train_steps": resume_state["train_steps"],
        "log_steps": 0,
        "running_loss": 0,
        "running_sit_loss": 0,
        "running_student_loss": 0,
        "running_sra_loss": 0,
        "start_time": time(),
    }

    maybe_run_pending_linear_eval_on_resume(args, state, resume_state)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(state["start_epoch"], args.epochs):
        train_one_epoch(args, epoch, loader, sampler, state)

    model.eval()
    if student is not None:
        student.eval()
        projector.eval()
    if align_head is not None:
        align_head.eval()
    if repa_projector is not None:
        repa_projector.eval()
    if sra_head is not None:
        sra_head.eval()
    logger.info("Done!")
    cleanup()


def add_data_args(parser):
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="ImageNet train root: local CLS-LOC/train dir or s3://.../CLS-LOC/train prefix.",
    )
    parser.add_argument(
        "--s3-endpoint",
        type=str,
        default=None,
        help="Deprecated (ignored). S3 reads use Petrel SDK via PETRELOSS_CONF / cfs/petreloss.conf.",
    )
    parser.add_argument(
        "--s3-region",
        type=str,
        default="us-east-1",
        help="Deprecated (ignored). Kept for backward-compatible CLI.",
    )
    parser.add_argument(
        "--s3-index-cache",
        type=str,
        default=None,
        help="Optional shared filesystem path to a pickle index; rank 0 builds once, all ranks load (recommended for multi-GPU).",
    )
    parser.add_argument(
        "--petrel-conf",
        type=str,
        default=None,
        help="Path to petreloss.conf; default PETRELOSS_CONF or $ZWL_ROOT/cfs/petreloss.conf.",
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Optional root for model checkpoints (e.g. /tmp/sit_checkpoints). "
        "Saves to <ckpt-dir>/<results-dir-basename>/<experiment>/checkpoints. "
        "Eval metrics/logs still go under --results-dir.",
    )
    parser.add_argument(
        "--ckpt-rclone-remote",
        type=str,
        default=None,
        help="Optional rclone remote root (e.g. tf:gugroup/zwl). After each checkpoint save, "
        "upload the file to <remote>/<results-run-tag>/<experiment>/checkpoints/<file> via rclone copyto, "
        "mirroring the path under --ckpt-dir.",
    )
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")


def add_model_args(parser):
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--uncondition", action="store_true")


def add_training_args(parser):
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout-minutes",
        type=float,
        default=180.0,
        help=(
            "NCCL collective timeout in minutes. Must exceed the rank-0-only FID "
            "evaluator runtime, otherwise idle ranks trip the watchdog and abort "
            "the run at the eval stage."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="Optional positive step cap for smoke tests; 0 means full epochs.",
    )
    parser.add_argument("--ckpt-every-epoch", type=int, default=40)
    parser.add_argument(
        "--eval-during-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run periodic FID evaluation during training (default: enabled).",
    )
    parser.add_argument("--eval-num-fid-samples", type=int, default=50_000)
    parser.add_argument("--eval-per-proc-batch-size", type=int, default=4)
    parser.add_argument("--eval-cfg-scale", type=float, default=1.0)
    parser.add_argument("--eval-num-sampling-steps", type=int, default=250)
    parser.add_argument("--eval-sampling-method", type=str, default="dopri5")
    parser.add_argument("--eval-atol", type=float, default=1e-6)
    parser.add_argument("--eval-rtol", type=float, default=1e-3)
    parser.add_argument(
        "--eval-temp-root",
        type=str,
        default=None,
        help="Optional root directory for temporary eval samples/npz. Metrics remain in results_dir.",
    )
    parser.add_argument(
        "--fid-ref-path",
        type=str,
        default=None,
        help="Reference batch npz path used by guided-diffusion evaluator.py.",
    )
    parser.add_argument(
        "--evaluator-script",
        type=str,
        default=None,
        help="Path to guided-diffusion evaluations/evaluator.py.",
    )
    parser.add_argument("--eval-log-interval", type=int, default=20)
    parser.add_argument("--keep-eval-samples", action="store_true")
    parser.add_argument(
        "--linear-eval-during-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run linear probe eval on frozen student at each --ckpt-every-epoch interval "
        "(only when --uncondition and student distillation is enabled).",
    )
    parser.add_argument(
        "--linear-train-data-path",
        type=str,
        default=None,
        help="S3 prefix for linear probe training set (s3://.../train).",
    )
    parser.add_argument(
        "--linear-val-data-path",
        type=str,
        default=None,
        help="Image root for linear probe validation set.",
    )
    parser.add_argument(
        "--linear-train-label-file",
        type=str,
        default="",
        help="Label file for linear probe training set.",
    )
    parser.add_argument(
        "--linear-val-label-file",
        type=str,
        default="",
        help="Label file for linear probe validation set.",
    )
    parser.add_argument("--linear-epochs", type=int, default=30)
    parser.add_argument(
        "--linear-eval-start-epoch",
        type=int,
        default=1,
        help="First training epoch (1-based) at which linear probe eval may run.",
    )
    parser.add_argument("--linear-global-batch-size", type=int, default=1024)
    parser.add_argument("--linear-lr", type=float, default=1e-3)
    parser.add_argument("--linear-num-workers", type=int, default=8)
    parser.add_argument(
        "--linear-use-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AMP during linear probe feature extraction.",
    )
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None, help="Backward-compatible alias for resume.")
    parser.add_argument(
        "--freeze-generator",
        action="store_true",
        help="Freeze the SiT generator and its EMA after resuming; train student and prediction heads only.",
    )
    parser.add_argument(
        "--freeze-student",
        action="store_true",
        help="Freeze the student encoder and its EMA after resuming; train the generator and mapper heads.",
    )
    parser.add_argument("--bf16", action="store_true")


def add_student_args(parser):
    parser.add_argument("--student-depth", type=int, default=4)
    parser.add_argument("--student-hidden-size", type=int, default=384)
    parser.add_argument("--student-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--student-feature-layer", type=int, default=-1)
    parser.add_argument(
        "--student-feature-layers",
        type=str,
        default=None,
        help="Comma-separated SiT block indices (after each block) for teacher features, "
        "concat on last dim, e.g. '4,8,11'. Negative indices allowed. When set, overrides --student-feature-layer.",
    )
    parser.add_argument(
        "--student-loss-weight",
        type=float,
        default=1.0,
        help="Distillation weight; use 0 for SiT-only baseline (student and projector are not instantiated).",
    )
    parser.add_argument("--student-mask-ratio", type=float, default=0.75)
    parser.add_argument("--student-use-mask", action="store_true")
    parser.add_argument("--student-direct-image-input", action="store_true")
    parser.add_argument(
        "--student-align-cosine-weight",
        type=float,
        default=0.0,
        help="Token-level cosine align: clean unmasked student (no grad, detached) vs align_head(teacher_align). "
        "Teacher single forward returns concat teacher_feat plus an optional align layer; align_head maps "
        "teacher H -> student D. Student backbone frozen for this term.",
    )
    parser.add_argument(
        "--student-ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Maintain an EMA copy of the student. Used for linear probe eval and saved in checkpoints. "
        "When --student-align-cosine-weight > 0, the EMA student is also used for the clean unmasked forward.",
    )
    parser.add_argument(
        "--student-clean-ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alias for --student-ema (kept for backward compatibility).",
    )
    parser.add_argument(
        "--student-align-teacher-layer",
        type=int,
        default=None,
        help="SiT block index (after each block; negatives allowed) for align_head input. "
        "Need not be in --student-feature-layers. If unset, uses the first listed distill layer.",
    )
    parser.add_argument("--projector-hidden-size", type=int, default=384)
    parser.add_argument("--projector-depth", type=int, default=2)
    parser.add_argument("--projector-mlp-ratio", type=float, default=4.0)
    parser.add_argument(
        "--projector-num-layers",
        type=int,
        default=1,
        help="How many student blocks to concat on the channel dim as projector input. "
        "Layers are chosen uniformly and always include the last block "
        "(depth=24, n=4 → 5,11,17,23). Default 1 is last layer only.",
    )
    parser.add_argument(
        "--projector-feature-layers",
        type=str,
        default=None,
        help="Comma-separated student block indices (after each block) concatenated on "
        "the last dim as projector input, e.g. '5,11,17,23'. Negative indices allowed. "
        "The last student block is always included. When set, overrides --projector-num-layers.",
    )


def add_repa_args(parser):
    """Optional REPA (DINOv2) alignment. Default coeff=0 keeps old configs unchanged."""
    parser.add_argument(
        "--repa-proj-coeff",
        type=float,
        default=0.0,
        help="Weight on REPA neg-cosine loss (REPA-main proj_coeff). Default 0 disables DINOv2/projector.",
    )
    parser.add_argument(
        "--repa-enc-type",
        type=str,
        default="dinov2-vit-b",
        help="Vision encoder type, e.g. dinov2-vit-b / dinov2-vit-l / dinov2-vit-g.",
    )
    parser.add_argument(
        "--repa-encoder-depth",
        type=int,
        default=8,
        help="1-based SiT block depth for REPA (REPA-main default 8 → 0-based index 7).",
    )
    parser.add_argument(
        "--repa-projector-dim",
        type=int,
        default=2048,
        help="Hidden width of the 3-layer REPA MLP projector.",
    )
    parser.add_argument(
        "--repa-encoder-repo",
        type=str,
        default=None,
        help="Optional local path to facebookresearch/dinov2 repo for torch.hub local load.",
    )
    parser.add_argument(
        "--repa-encoder-ckpt",
        type=str,
        default=None,
        help="Optional local DINOv2 checkpoint (.pth). If unset, hub pretrained weights are used.",
    )


def add_sra_args(parser):
    """Optional SRA (self-representation alignment). Default weight=0 keeps existing scripts unchanged."""
    parser.add_argument(
        "--sra-loss-weight",
        type=float,
        default=0.0,
        help="Weight on SRA alignment. Default 0 disables SRA (no extra head / EMA feature forward). "
        "SiT-SRA cosine uses 0.5. Requires --student-loss-weight 0.",
    )
    parser.add_argument(
        "--sra-block-out-s",
        type=int,
        default=8,
        help="1-based earlier block for the online model (paper SiT-XL/2: 8 → 0-based 7).",
    )
    parser.add_argument(
        "--sra-block-out-t",
        type=int,
        default=20,
        help="1-based later block for the EMA teacher (paper SiT-XL/2: 20 → 0-based 19).",
    )
    parser.add_argument(
        "--sra-t-max",
        type=float,
        default=0.2,
        help="Max time gap between online and EMA. This codebase uses t=0 noise / t=1 data, "
        "so the EMA teacher is sampled at t + t_max * U(0,1) (cleaner).",
    )
    parser.add_argument(
        "--sra-loss-type",
        type=str,
        default="cos",
        choices=["cos", "sml1", "l2", "l1"],
        help="SRA alignment loss. Paper reproduction used sml1; README prefers cosine.",
    )


def build_parser():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_model_args(parser)
    add_training_args(parser)
    add_student_args(parser)
    add_repa_args(parser)
    add_sra_args(parser)
    parse_transport_args(parser)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
