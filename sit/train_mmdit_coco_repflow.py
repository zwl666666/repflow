"""Train COCO MMDiT with native flow matching or MMDiT + RepFlow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

from coco_text_dataset import CocoTextDataset
from coco_samplers import euler_sampler
from evaluate_coco_mmdit import compute_metrics
from mmdit_coco import MMDiT
from repflow_student import build_repflow_modules
from sample_coco_mmdit import build_manifest, noise_from_seeds


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_L14 = PROJECT_ROOT / "hf_models" / "clip-vit-large-patch14"
DEFAULT_CLIP_B32 = PROJECT_ROOT / "hf_models" / "clip-vit-base-patch32"
DEFAULT_VAE = PROJECT_ROOT / "hf_models" / "sd-vae-ft-ema"
DEFAULT_GENERATOR_INIT = Path(__file__).resolve().parent / "runs" / "mmdit_coco_generator_init.pt"


def requires_grad(model, flag=True):
    for parameter in model.parameters():
        parameter.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter.data, alpha=1 - decay)


def token_level_cosine_loss(teacher_tokens, student_tokens, eps=1e-8):
    teacher = F.normalize(teacher_tokens, dim=-1, eps=eps)
    student = F.normalize(student_tokens, dim=-1, eps=eps)
    return (1.0 - (teacher * student).sum(dim=-1)).mean()


def masked_smooth_l1_loss(pred, target, mask=None, eps=1e-6):
    loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    if mask is None:
        return loss.mean()
    mask = mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(eps)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["native", "repflow"], required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--captions-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=150000)
    parser.add_argument("--checkpoint-every", type=int, default=50000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout-minutes",
        type=float,
        default=180.0,
        help=(
            "NCCL/Accelerate collective timeout in minutes. Must exceed the "
            "rank-0-heavy FID evaluator runtime during periodic evaluation."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--caption-dropout", type=float, default=0.1)
    parser.add_argument(
        "--train-context-file",
        type=str,
        default=None,
        help="Optional precomputed CLIP context file for the training set.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=str(DEFAULT_CLIP_L14),
    )
    parser.add_argument("--vae-model", type=str, default=str(DEFAULT_VAE))
    parser.add_argument("--latent-scale", type=float, default=0.18215)
    parser.add_argument("--generator-ema-decay", type=float, default=0.9999)
    parser.add_argument("--student-ema-decay", type=float, default=0.9999)
    parser.add_argument("--g2r-layers", type=str, default="4,10,16,22")
    parser.add_argument("--r2g-layer", type=int, default=7)
    parser.add_argument("--lambda-g2r", type=float, default=1.0)
    parser.add_argument("--lambda-r2g", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--student-depth", type=int, default=24)
    parser.add_argument("--student-hidden-size", type=int, default=1024)
    parser.add_argument("--student-heads", type=int, default=16)
    parser.add_argument("--predictor-depth", type=int, default=10)
    parser.add_argument("--predictor-hidden-size", type=int, default=384)
    parser.add_argument("--predictor-heads", type=int, default=12)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--reset-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load model weights from --resume but start a fresh optimizer state.",
    )
    parser.add_argument("--save-every", type=int, default=50000)
    parser.add_argument(
        "--generator-init",
        type=str,
        default=str(DEFAULT_GENERATOR_INIT),
        help="Shared random-initialized generator state used by both native and RepFlow runs.",
    )
    parser.add_argument("--eval-during-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-every", type=int, default=50000)
    parser.add_argument("--eval-num-images", type=int, default=30000)
    parser.add_argument("--eval-image-dir", type=str, default=None)
    parser.add_argument("--eval-captions-file", type=str, default=None)
    parser.add_argument("--eval-output-dir", type=str, default=None)
    parser.add_argument(
        "--eval-assets-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "eval_assets" / "coco_val2014_30k"),
    )
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-steps", type=int, default=50)
    parser.add_argument("--eval-cfg-scale", type=float, default=2.0)
    parser.add_argument(
        "--eval-clip-model",
        type=str,
        default=str(DEFAULT_CLIP_B32),
    )
    parser.add_argument("--eval-num-workers", type=int, default=8)
    args = parser.parse_args()
    args.g2r_layers = tuple(int(x) for x in args.g2r_layers.split(","))
    return args


def build_optimizer(args, model, student, predictor, r2g_projector):
    groups = [
        {
            "params": list(model.parameters()),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
            "name": "generator",
        }
    ]
    if args.method == "repflow":
        groups.append(
            {
                "params": list(student.parameters()) + list(predictor.parameters()),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                "name": "student_predictor",
            }
        )
        groups.append(
            {
                "params": list(r2g_projector.parameters()),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                "name": "r2g_projector",
            }
        )
    return torch.optim.AdamW(groups)


def clip_optimizer_groups(accelerator, optimizer, max_norm):
    norms = []
    for group in optimizer.param_groups:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            norms.append(0.0)
            continue
        norm = accelerator.clip_grad_norm_(params, max_norm)
        norms.append(float(norm))
    return norms


def apply_learning_rate(optimizer, args, step, warmup_origin):
    if args.lr_warmup_steps > 0:
        progress = max(0, step - warmup_origin)
        scale = min(1.0, (progress + 1) / args.lr_warmup_steps)
    else:
        scale = 1.0
    learning_rate = args.lr * scale
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def save_checkpoint(path, args, step, model, generator_ema, student, student_ema, projector, r2g_projector, optimizer):
    payload = {
        "step": step,
        "args": vars(args),
        "model": model.state_dict(),
        "generator_ema": generator_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if args.method == "repflow":
        payload.update(
            {
                "student": student.state_dict(),
                "student_ema": student_ema.state_dict(),
                "projector": projector.state_dict(),
                "r2g_projector": r2g_projector.state_dict(),
            }
        )
    torch.save(payload, path)


def run_training_eval(
    args,
    accelerator,
    generator_ema,
    tokenizer,
    text_encoder,
    vae,
    step,
    device,
):
    if not args.eval_during_train:
        return
    if step % args.eval_every != 0:
        return
    if not args.eval_image_dir or not args.eval_captions_file:
        raise ValueError("--eval-during-train requires --eval-image-dir and --eval-captions-file")

    eval_root = Path(args.eval_output_dir or Path(args.output_dir) / "eval")
    stage_dir = eval_root / f"step_{step:07d}"
    images_dir = stage_dir / "images"
    manifest_path = stage_dir / "manifest.jsonl"
    metrics_path = stage_dir / "metrics.json"

    if accelerator.is_main_process:
        eval_root.mkdir(parents=True, exist_ok=True)
        stage_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(args.eval_captions_file, args.eval_num_images, 2027)
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in manifest:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    accelerator.wait_for_everyone()

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]

    generator_ema.eval()
    empty_tokens = tokenizer(
        [""],
        truncation=True,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    )
    empty_context = text_encoder(
        input_ids=empty_tokens.input_ids.to(device),
        attention_mask=empty_tokens.attention_mask.to(device),
    ).last_hidden_state

    total = len(manifest)
    stride = args.eval_batch_size * accelerator.num_processes
    first_start = accelerator.process_index * args.eval_batch_size
    for start in range(first_start, total, stride):
        batch = manifest[start : start + args.eval_batch_size]
        captions = [record["caption"] for record in batch]
        seeds = [record["seed"] for record in batch]
        tokens = tokenizer(
            captions,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        context = text_encoder(
            input_ids=tokens.input_ids.to(device),
            attention_mask=tokens.attention_mask.to(device),
        ).last_hidden_state
        noise = noise_from_seeds(seeds, (4, 32, 32), device)
        current_null_context = empty_context.expand(context.shape[0], -1, -1)

        with torch.no_grad(), accelerator.autocast():
            latents = euler_sampler(
                generator_ema,
                noise,
                context,
                current_null_context,
                num_steps=args.eval_num_steps,
                cfg_scale=args.eval_cfg_scale,
            )
            decoded = vae.decode(latents / args.latent_scale).sample

        images = ((decoded.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        for offset, array in enumerate(images):
            image_index = start + offset
            Image.fromarray(array).save(images_dir / f"{image_index:08d}.png")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metrics = compute_metrics(
            images_dir,
            args.eval_image_dir,
            args.eval_captions_file,
            manifest_path,
            device,
            batch_size=args.eval_batch_size,
            clip_model=args.eval_clip_model,
            num_workers=args.eval_num_workers,
            fid_stats=str(Path(args.eval_assets_dir) / "fid_stats.npz")
            if Path(args.eval_assets_dir, "fid_stats.npz").exists()
            else None,
            clip_text_features=torch.from_numpy(
                np.load(Path(args.eval_assets_dir) / "clip_text_features.npy")
            )
            if Path(args.eval_assets_dir, "clip_text_features.npy").exists()
            else None,
            fid_graph=str(Path(__file__).resolve().parent / "classify_image_graph_def.pb"),
        )
        metrics["step"] = step
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        print(f"Saved eval metrics to {metrics_path}")
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    if args.method == "native" and (args.lambda_g2r != 0 or args.lambda_r2g != 0):
        raise ValueError("Native training must set both RepFlow loss weights to 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=output_dir),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False),
            InitProcessGroupKwargs(
                backend="nccl",
                timeout=timedelta(minutes=args.dist_timeout_minutes),
            )
        ],
    )
    world_size = accelerator.num_processes
    if args.micro_batch_size * world_size * args.gradient_accumulation_steps != args.global_batch_size:
        raise ValueError(
            "micro_batch_size * world_size * gradient_accumulation_steps must equal global_batch_size"
        )
    set_seed(args.seed)
    device = accelerator.device

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    text_encoder = CLIPTextModel.from_pretrained(args.clip_model).to(device)
    requires_grad(text_encoder, False)
    text_encoder.eval()

    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    requires_grad(vae, False)
    vae.eval()

    dataset = CocoTextDataset(
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        image_size=args.resolution,
        caption_dropout=args.caption_dropout,
        seed=args.seed,
        context_file=args.train_context_file,
    )
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    latent_size = args.resolution // 8
    model = MMDiT(
        input_size=latent_size,
        patch_size=2,
        in_channels=4,
        depth=24,
        mlp_ratio=4.0,
        context_dim=768,
        qkv_bias=True,
        learn_sigma=False,
    ).to(device)
    generator_hidden_size = model.hidden_size

    generator_init_path = Path(args.generator_init)
    if not args.resume:
        if accelerator.is_main_process:
            generator_init_path.parent.mkdir(parents=True, exist_ok=True)
            if not generator_init_path.exists():
                temporary_path = generator_init_path.with_suffix(generator_init_path.suffix + ".tmp")
                torch.save(model.state_dict(), temporary_path)
                os.replace(temporary_path, generator_init_path)
        accelerator.wait_for_everyone()
        generator_init_state = torch.load(generator_init_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            generator_init_state,
            strict=False,
        )
        if missing_keys:
            raise RuntimeError(
                f"Shared generator init is missing required keys: {missing_keys[:8]}"
            )

    generator_ema = deepcopy(model).to(device)
    requires_grad(generator_ema, False)
    generator_ema.eval()

    student = student_ema = projector = r2g_projector = None
    if args.method == "repflow":
        student, projector, r2g_projector = build_repflow_modules(
            image_size=args.resolution,
            student_depth=args.student_depth,
            student_hidden_size=args.student_hidden_size,
            student_heads=args.student_heads,
            mask_ratio=args.mask_ratio,
            predictor_depth=args.predictor_depth,
            predictor_hidden_size=args.predictor_hidden_size,
            predictor_heads=args.predictor_heads,
            teacher_hidden_size=model.hidden_size * len(args.g2r_layers),
            generator_hidden_size=model.hidden_size,
            device=device,
        )
        student_ema = deepcopy(student).to(device)
        requires_grad(student_ema, False)
        student_ema.eval()
        student.train()
        projector.train()
        r2g_projector.train()

    optimizer = build_optimizer(args, model, student, projector, r2g_projector)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if args.method == "repflow":
        student, projector, r2g_projector = accelerator.prepare(student, projector, r2g_projector)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
        generator_ema.load_state_dict(checkpoint["generator_ema"])
        if not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if args.method == "repflow":
            accelerator.unwrap_model(student).load_state_dict(checkpoint["student"])
            student_ema.load_state_dict(checkpoint["student_ema"])
            accelerator.unwrap_model(projector).load_state_dict(checkpoint["projector"])
            accelerator.unwrap_model(r2g_projector).load_state_dict(checkpoint["r2g_projector"])

    model.train()
    start_step = 0
    if args.resume:
        start_step = int(checkpoint["step"])
    warmup_origin = start_step

    log_path = output_dir / "training_log.csv"
    if not log_path.exists() and accelerator.is_main_process:
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step",
                    "loss_fm",
                    "loss_g2r",
                    "loss_r2g",
                    "lr",
                    "elapsed_sec",
                ]
            )

    if sampler is not None:
        sampler.set_epoch(start_step // max(1, len(dataloader)))

    start_time = time.time()
    step = start_step
    while step < args.max_train_steps:
        for images, texts in dataloader:
            if step >= args.max_train_steps:
                break
            images = images.to(device, non_blocking=True)
            if isinstance(texts, torch.Tensor):
                context = texts.to(device, non_blocking=True)
            else:
                tokenized = tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt",
                )
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)
                context = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state

            with torch.no_grad(), accelerator.autocast():
                posterior = vae.encode(images).latent_dist
                mean = posterior.mean
                logvar = posterior.logvar.clamp(-30.0, 20.0)
                z0 = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
                z0 = z0 * args.latent_scale

            batch_size = z0.shape[0]
            t = torch.rand(batch_size, device=device)
            noise = torch.randn_like(z0)
            expanded_t = t.view(-1, 1, 1, 1)
            z_t = (1.0 - expanded_t) * z0 + expanded_t * noise
            target = noise - z0

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    if args.method == "native":
                        pred, _, _ = model(z_t, t, context, return_features=False)
                        fm_loss = F.mse_loss(pred, target, reduction="mean")
                        g2r_loss = torch.zeros((), device=device)
                        r2g_loss = torch.zeros((), device=device)
                    else:
                        pred, g2r_features, r2g_feature = model(
                            z_t,
                            t,
                            context,
                            return_features=True,
                            g2r_layers=args.g2r_layers,
                            r2g_layer=args.r2g_layer,
                        )
                        fm_loss = F.mse_loss(pred, target, reduction="mean")

                        normalized = [
                            F.layer_norm(g2r_features[layer], (generator_hidden_size,))
                            for layer in args.g2r_layers
                        ]
                        g2r_target = torch.cat(normalized, dim=-1).detach()
                        student_tokens, mask, keep_idx = student(
                            images,
                            use_mask=True,
                            return_mask=True,
                        )
                        predicted_g2r = projector(student_tokens, t, keep_idx)
                        g2r_loss = masked_smooth_l1_loss(predicted_g2r, g2r_target, mask)

                        with torch.no_grad():
                            student_clean = student_ema(images, use_mask=False, return_mask=False)
                        projected_r2g = r2g_projector(r2g_feature)
                        r2g_loss = token_level_cosine_loss(projected_r2g, student_clean.detach())

                    loss = fm_loss + args.lambda_g2r * g2r_loss + args.lambda_r2g * r2g_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    apply_learning_rate(optimizer, args, step, warmup_origin)
                    clip_optimizer_groups(accelerator, optimizer, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(generator_ema, accelerator.unwrap_model(model), args.generator_ema_decay)
                    if args.method == "repflow":
                        update_ema(student_ema, accelerator.unwrap_model(student), args.student_ema_decay)
                    step += 1

                    if accelerator.is_main_process and step % args.log_every == 0:
                        elapsed = time.time() - start_time
                        lr = optimizer.param_groups[0]["lr"]
                        with log_path.open("a", encoding="utf-8", newline="") as handle:
                            writer = csv.writer(handle)
                            writer.writerow(
                                [
                                    step,
                                    float(fm_loss.detach().item()),
                                    float(g2r_loss.detach().item()),
                                    float(r2g_loss.detach().item()),
                                    float(lr),
                                    elapsed,
                                ]
                            )
                        print(
                            f"step={step} loss_fm={float(fm_loss.detach()):.6f} "
                            f"loss_g2r={float(g2r_loss.detach()):.6f} "
                            f"loss_r2g={float(r2g_loss.detach()):.6f} elapsed={elapsed:.1f}s"
                        )

                    if accelerator.is_main_process and step % args.save_every == 0:
                        path = checkpoint_dir / f"step_{step:07d}.pt"
                        save_checkpoint(
                            path,
                            args,
                            step,
                            accelerator.unwrap_model(model),
                            generator_ema,
                            accelerator.unwrap_model(student) if student is not None else None,
                            student_ema,
                            accelerator.unwrap_model(projector) if projector is not None else None,
                            accelerator.unwrap_model(r2g_projector) if r2g_projector is not None else None,
                            optimizer,
                        )
                        print(f"Saved checkpoint to {path}")

                    if step % args.eval_every == 0:
                        run_training_eval(
                            args,
                            accelerator,
                            generator_ema,
                            tokenizer,
                            text_encoder,
                            vae,
                            step,
                            device,
                        )

        if sampler is not None:
            sampler.set_epoch(sampler.epoch + 1)

    if accelerator.is_main_process and (step % args.save_every != 0):
        path = checkpoint_dir / f"step_{step:07d}.pt"
        save_checkpoint(
            path,
            args,
            step,
            accelerator.unwrap_model(model),
            generator_ema,
            accelerator.unwrap_model(student) if student is not None else None,
            student_ema,
            accelerator.unwrap_model(projector) if projector is not None else None,
            accelerator.unwrap_model(r2g_projector) if r2g_projector is not None else None,
            optimizer,
        )
        print(f"Saved final checkpoint to {path}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
