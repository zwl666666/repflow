"""Training step for JiT FD post-training."""

from __future__ import annotations

import logging
import math

import torch

from frechet_distance.judges_jit import extract_judge_features
from frechet_distance.losses import compute_frechet_distance_loss, diff_all_gather
from util import misc

logger = logging.getLogger("jit_fd")


def adjust_lr(optimizer, step: int, args):
    if step < args.fd_warmup_steps:
        lr = args.lr * step / max(1, args.fd_warmup_steps)
    elif args.lr_schedule == "constant":
        lr = args.lr
    elif args.lr_schedule == "cosine":
        progress = (step - args.fd_warmup_steps) / max(1, args.total_steps - args.fd_warmup_steps)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise NotImplementedError(args.lr_schedule)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def get_fd_train_step(denoiser, judges, args):
    fid_norm_eps = args.fd_fid_norm_eps
    batch_size = args.batch_size
    num_classes = args.class_num
    num_steps = args.fd_num_sampling_steps
    cfg_scale = args.fd_train_cfg

    def fd_train_step():
        labels = torch.randint(0, num_classes, (batch_size,), device="cuda")
        sampled = denoiser.sample_images_with_grad(
            labels,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            cfg_interval=(args.interval_min, args.interval_max),
        )

        loss = torch.tensor(0.0, device="cuda")
        loss_dict = {}
        all_new_feats = []

        for judge in judges:
            feats = extract_judge_features(judge, sampled, labels)
            new_feats = diff_all_gather(feats)
            all_new_feats.append(new_feats)

        for i, judge in enumerate(judges):
            new_feats = all_new_feats[i]
            ns_kwargs = dict(sigma_ref_sqrt=judge.get("sigma_ref_sqrt"))
            if judge["queue"].online_accum or judge["queue"].ema_stats:
                mu, sigma = judge["queue"].build_feats_stats(new_feats)
                fid = compute_frechet_distance_loss(
                    judge["mu_ref"],
                    judge["sigma_ref"],
                    mu=mu,
                    sigma=sigma,
                    **ns_kwargs,
                )
            else:
                all_feats = judge["queue"].build_feats_snapshot(new_feats)
                fid = compute_frechet_distance_loss(
                    judge["mu_ref"],
                    judge["sigma_ref"],
                    all_feats=all_feats,
                    **ns_kwargs,
                )
            fid_loss = fid / (fid.detach() + fid_norm_eps)
            loss = loss + judge["weight"] * fid_loss
            loss_dict[f"fd_{judge['name']}"] = float(fid.detach())

        loss.backward()

        if misc.is_dist_avail_and_initialized():
            for p in denoiser.net.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        for i, judge in enumerate(judges):
            judge["queue"].enqueue(all_new_feats[i].detach())

        return loss, loss_dict

    return fd_train_step
