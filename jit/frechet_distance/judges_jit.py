"""Judge / queue helpers for JiT FD post-training."""

from __future__ import annotations

import logging

import numpy as np
import torch

from frechet_distance.losses import compute_frechet_distance_loss, diff_all_gather
from frechet_distance.metrics import compute_fid as np_fid
from frechet_distance.queue import FeatureQueue
from frechet_distance.student_repr import StudentReprModel
from util import misc

logger = logging.getLogger("jit_fd")


def _extract_judge_features_impl(
    judge: dict, images: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return judge["model"](images, labels)


# Student blocks use @torch.compile; after FDr-6 eval the CUDA allocator is
# fragmented and inductor may request large contiguous buffers. Eager mode is
# enough for the frozen judge forward and avoids post-eval OOM.
# torch.compiler.disable must be used as a decorator, not a context manager.
_dynamo_disable = getattr(torch.compiler, "disable", None)
if _dynamo_disable is None:
    _dynamo_disable = getattr(torch._dynamo, "disable", None)

if _dynamo_disable is not None:
    extract_judge_features = _dynamo_disable(_extract_judge_features_impl)
else:
    extract_judge_features = _extract_judge_features_impl


def build_student_judge(
    repr_model: StudentReprModel,
    ref_stats_path: str,
    *,
    queue_size: int,
    fd_online_accum: bool,
    fd_ema_beta: float,
    fd_eigvalsh: bool,
    weight: float = 1.0,
    name: str = "student",
) -> dict:
    from frechet_distance.losses import load_mu_and_sigma_reference, precompute_sigma_ref_sqrt

    mu_ref, sigma_ref = load_mu_and_sigma_reference(ref_stats_path)
    queue = FeatureQueue(
        size=queue_size,
        feat_dim=repr_model.feat_dim,
        online_accum=fd_online_accum,
        ema_beta=fd_ema_beta,
    ).cuda()

    sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref) if fd_eigvalsh else None
    stats_mode = (
        f"ema(beta={fd_ema_beta})"
        if fd_ema_beta > 0
        else ("online_accum" if fd_online_accum else "snapshot")
    )
    logger.info(
        "[FD] Student judge (%s): feat_dim=%d weight=%.3f stats=%s stats_mode=%s eigvalsh=%s",
        name,
        repr_model.feat_dim,
        weight,
        ref_stats_path,
        stats_mode,
        fd_eigvalsh,
    )
    return {
        "name": name,
        "model": repr_model,
        "feat_dim": repr_model.feat_dim,
        "mu_ref": mu_ref,
        "sigma_ref": sigma_ref,
        "sigma_ref_sqrt": sigma_ref_sqrt,
        "queue": queue,
        "weight": weight,
    }


def save_fd_queue_states(judges):
    return [{"name": j["name"], "queue": j["queue"].state_dict()} for j in judges]


def load_fd_queue_states(judges, saved_states) -> bool:
    name_to_state = {s["name"]: s["queue"] for s in saved_states}
    loaded = 0
    for judge in judges:
        if judge["name"] in name_to_state:
            judge["queue"].load_state_dict(name_to_state[judge["name"]])
            judge["queue"].cuda()
            loaded += 1
            logger.info("[FD] Restored queue state for '%s'", judge["name"])
        else:
            logger.warning("[FD] No saved queue state for '%s'", judge["name"])
    return loaded == len(judges)


@torch.no_grad()
def fill_all_queues(judges, denoiser, args):
    queue_size = args.queue_size
    if queue_size == 0:
        logger.info("[FD] queue_size=0: skipping queue fill")
        return

    was_training = denoiser.training
    denoiser.eval()
    filled = 0
    num_classes = args.class_num

    while filled < queue_size:
        batch_size = min(args.fd_queue_fill_bsz, queue_size - filled)
        labels = torch.randint(0, num_classes, (batch_size,), device="cuda")
        saved_steps = denoiser.steps
        denoiser.steps = getattr(args, "fd_num_sampling_steps", saved_steps)
        try:
            imgs = denoiser.generate(labels)
        finally:
            denoiser.steps = saved_steps

        for judge in judges:
            student_dev = next(judge["model"].net_student.parameters()).device
            imgs_in = imgs.to(student_dev, non_blocking=True)
            labels_in = labels.to(student_dev, non_blocking=True)
            local_feats = extract_judge_features(judge, imgs_in, labels_in)
            if local_feats.device.type != "cuda":
                local_feats = local_feats.cuda(non_blocking=True)
            all_feats = diff_all_gather(local_feats)
            count = min(all_feats.shape[0], queue_size - filled)
            q = judge["queue"]
            if q.ema_stats:
                q.accumulate_batch(all_feats[:count])
            else:
                q.feats[filled : filled + count] = all_feats[:count].float()
        filled += count
        if misc.is_main_process():
            logger.info("[FD] Queue fill: %d/%d (%.1f%%)", filled, queue_size, filled / queue_size * 100)

    for judge in judges:
        q = judge["queue"]
        if q.ema_stats:
            q._finalize_streaming_init()
        else:
            q.ptr.zero_()
            if q.online_accum:
                q._init_accumulators()

    if was_training:
        denoiser.train()
    logger.info("[FD] Queue initialized with %d features", filled)


@torch.no_grad()
def run_sanity_check(judges, queue_size, args=None):
    if not misc.is_main_process() or queue_size == 0:
        if queue_size == 0:
            logger.info("[FD] queue_size=0: skipping sanity check")
        return

    use_eigvalsh = args.fd_eigvalsh if args else False

    for judge in judges:
        q = judge["queue"]
        if q.ema_stats:
            sigma_ema = q.m2_ema - q.mu_ema.unsqueeze(1) * q.mu_ema.unsqueeze(0)
            fid_ema = float(
                compute_frechet_distance_loss(
                    judge["mu_ref"],
                    judge["sigma_ref"],
                    mu=q.mu_ema,
                    sigma=sigma_ema,
                ).item()
            )
            logger.info("[FD] Sanity '%s': FID=%.4f (ema_init)", judge["name"], fid_ema)
            continue

        feats_np = q.feats.cpu().float().numpy()
        fid_numpy = np_fid(
            np.mean(feats_np, 0),
            np.cov(feats_np, rowvar=False),
            judge["mu_ref"].cpu().numpy(),
            judge["sigma_ref"].cpu().numpy(),
        )
        fid_diff = compute_frechet_distance_loss(
            judge["mu_ref"],
            judge["sigma_ref"],
            all_feats=q.feats,
        ).item()
        parts = [f"FID={fid_numpy:.4f} (numpy)", f"FID={fid_diff:.4f} (diff/eigvals)"]
        sigma_ref_sqrt = judge.get("sigma_ref_sqrt")
        if use_eigvalsh and sigma_ref_sqrt is not None:
            fid_eigvalsh = compute_frechet_distance_loss(
                judge["mu_ref"],
                judge["sigma_ref"],
                all_feats=q.feats,
                sigma_ref_sqrt=sigma_ref_sqrt,
            ).item()
            parts.append(f"FID={fid_eigvalsh:.4f} (eigvalsh, err={abs(fid_eigvalsh - fid_diff):.2e})")
        logger.info("[FD] Sanity '%s': %s", judge["name"], ", ".join(parts))
