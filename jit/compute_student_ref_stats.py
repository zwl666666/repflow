"""Compute reference mu/sigma for student representation on ImageNet train."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

from frechet_distance.student_repr import build_student_repr_model
from train_utils import (
    apply_ema_weights_from_checkpoint,
    build_denoiser,
    denoiser_load_exclude_prefixes,
    load_denoiser_state_dict,
)
from util import misc
from util.crop import center_crop_arr

logger = logging.getLogger("jit_fd")


def parse_args():
    parser = argparse.ArgumentParser("Compute student repr reference stats for FD loss")
    parser.add_argument("--load_from", required=True, type=str, help="Distillation checkpoint (.pth)")
    parser.add_argument("--args_json", default=None, type=str, help="Architecture args.json (default: sibling of ckpt)")
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Cap number of images (sequential ImageFolder order). Default: use full train set.",
    )
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--output_path", required=True, type=str, help="Output .npz path")
    parser.add_argument("--fd_student_t", type=float, default=1.0)
    parser.add_argument("--fd_use_projector", action="store_true")
    parser.add_argument("--fd_student_use_mask", action="store_true")
    parser.add_argument(
        "--no_fd_student_apply_norm",
        action="store_false",
        dest="fd_student_apply_norm",
        help="Skip student final LayerNorm when computing ref stats.",
    )
    parser.set_defaults(fd_student_apply_norm=True)
    parser.add_argument("--fd_student_pool", default="mean", choices=["mean", "sum", "flatten"])
    parser.add_argument(
        "--fd_use_ema",
        action="store_true",
        help="Load net + net_student from checkpoint model_ema before extracting ref stats.",
    )
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser.parse_args()


def load_architecture_args(args_json: str):
    with open(args_json, encoding="utf-8") as f:
        payload = json.load(f)
    ns = argparse.Namespace(**payload)
    ns.training_mode = "distill"
    return ns


@torch.inference_mode()
def main():
    args = parse_args()
    misc.init_distributed_mode(args)

    if args.args_json is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.load_from))
        candidate = os.path.join(ckpt_dir, "args.json")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"args.json not found next to checkpoint: {candidate}")
        args.args_json = candidate

    arch_args = load_architecture_args(args.args_json)
    arch_args.device = "cuda"
    arch_args.distributed = misc.is_dist_avail_and_initialized()
    arch_args.data_path = args.data_path
    arch_args.skip_projector = not args.fd_use_projector

    rank = misc.get_rank()
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)

    denoiser = build_denoiser(arch_args)
    checkpoint = torch.load(args.load_from, map_location="cpu", weights_only=False)
    exclude = denoiser_load_exclude_prefixes(arch_args)
    load_denoiser_state_dict(
        denoiser,
        checkpoint["model"],
        exclude_prefixes=exclude,
        strict=not exclude,
    )
    if args.fd_use_ema:
        source = apply_ema_weights_from_checkpoint(
            denoiser,
            checkpoint,
            skip_projector=arch_args.skip_projector,
            logger=logger if rank == 0 else None,
        )
        if rank == 0:
            logger.info("Ref stats: net + student from EMA (source=%s)", source)
    elif rank == 0:
        logger.info("Ref stats: online net + net_student from checkpoint['model']")
    denoiser.cuda().eval()
    del checkpoint

    repr_model = build_student_repr_model(
        denoiser,
        use_projector=args.fd_use_projector,
        fd_t=args.fd_student_t,
        use_mask=args.fd_student_use_mask,
        apply_final_norm=args.fd_student_apply_norm,
        pool=args.fd_student_pool,
    ).cuda()

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.PILToTensor(),
    ])
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"Local --data_path not found: {args.data_path}")
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(args.data_path, transform)
    total_available = len(dataset)
    if args.num_images is not None and args.num_images < total_available:
        dataset = torch.utils.data.Subset(dataset, list(range(args.num_images)))
        if rank == 0:
            logger.info("Using first %d / %d train images for ref stats", args.num_images, total_available)
    elif rank == 0:
        logger.info("Using full train set (%d images) for ref stats", total_available)

    sampler = (
        DistributedSampler(dataset, shuffle=False, drop_last=False)
        if misc.is_dist_avail_and_initialized()
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    feat_dim = repr_model.feat_dim
    feat_sum = torch.zeros(feat_dim, dtype=torch.float64, device="cuda")
    feat_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device="cuda")
    count = 0

    t0 = time.time()
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to("cuda", non_blocking=True).float().div_(255.0)
        images = images * 2.0 - 1.0
        labels = labels.to("cuda", non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = repr_model(images, labels).double()

        feat_sum.add_(feats.sum(0))
        feat_outer.addmm_(feats.T, feats)
        count += feats.shape[0]

        if rank == 0 and (batch_idx + 1) % 20 == 0:
            logger.info("Processed %d images ...", count)

    if misc.is_dist_avail_and_initialized():
        count_t = torch.tensor([count], dtype=torch.long, device="cuda")
        dist.reduce(feat_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(feat_outer, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)
        count = int(count_t.item())

    if rank != 0:
        return

    mu = (feat_sum / count).cpu().numpy()
    sigma = ((feat_outer - torch.outer(feat_sum, feat_sum) / count) / (count - 1)).cpu().numpy()
    np.savez(args.output_path, mu=mu.astype(np.float64), sigma=sigma.astype(np.float64), count=count)
    logger.info(
        "Saved student ref stats to %s (N=%d, dim=%d) in %.1fs",
        args.output_path,
        count,
        feat_dim,
        time.time() - t0,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
