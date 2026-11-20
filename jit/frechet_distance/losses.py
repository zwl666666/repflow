"""Differentiable FID computation for training-time loss."""

import logging

import numpy as np
import torch
import torch.distributed

logger = logging.getLogger("jit_fd")


class _DiffAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        world_size = torch.distributed.get_world_size()
        ctx.rank = torch.distributed.get_rank()
        ctx.batch_size = tensor.shape[0]
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor.contiguous())
        gathered[ctx.rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunk = ctx.batch_size
        return grad_output[ctx.rank * chunk : (ctx.rank + 1) * chunk].contiguous()


def diff_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        return tensor
    return _DiffAllGather.apply(tensor)


def precompute_sigma_ref_sqrt(sigma_ref: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(sigma_ref)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def _compute_trace_term(
    sigma: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if sigma_ref_sqrt is not None:
        m = sigma_ref_sqrt @ sigma @ sigma_ref_sqrt
        m = 0.5 * (m + m.T)
        evals = torch.linalg.eigvalsh(m)
        evals = torch.clamp(evals, min=0)
        tr_covmean = torch.sum(torch.sqrt(evals))
    else:
        product = sigma @ sigma_ref
        if not torch.isfinite(product).all():
            return None
        eigvals = torch.linalg.eigvals(product).real
        eigvals = torch.clamp(eigvals, min=0)
        tr_covmean = torch.sum(torch.sqrt(eigvals))

    return torch.diagonal(sigma).sum() + torch.diagonal(sigma_ref).sum() - 2.0 * tr_covmean


def compute_frechet_distance_loss(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    all_feats: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor:
    if all_feats is not None:
        n_samples = all_feats.shape[0]
        if n_samples < 2:
            logger.warning("[compute_frechet_distance_loss] Only %d sample(s) — need >= 2", n_samples)
            return torch.tensor(1e6, device=all_feats.device, dtype=torch.float32, requires_grad=True)
        mu = all_feats.mean(dim=0)
        feats_c = all_feats - mu
        sigma = (feats_c.T @ feats_c) / (n_samples - 1)
    elif mu is None or sigma is None:
        raise ValueError("Provide either all_feats or both mu and sigma")

    compute_dtype = sigma.dtype
    mu_ref = mu_ref.to(dtype=compute_dtype)
    sigma_ref = sigma_ref.to(dtype=compute_dtype)
    if sigma_ref_sqrt is not None:
        sigma_ref_sqrt = sigma_ref_sqrt.to(dtype=compute_dtype)

    diff = mu - mu_ref
    mean_term = diff.dot(diff)

    trace_term = _compute_trace_term(sigma, sigma_ref, sigma_ref_sqrt)
    if trace_term is None:
        device = all_feats.device if all_feats is not None else mu.device
        logger.warning("[compute_frechet_distance_loss] NaN/Inf in covariance product — returning fallback")
        return torch.tensor(1e6, device=device, dtype=torch.float32)

    return (mean_term + trace_term).float()


def load_mu_and_sigma_reference(fid_stats_path: str):
    ref = np.load(fid_stats_path)
    mu_ref = torch.tensor(ref["mu"], device="cuda", dtype=torch.float64)
    sigma_ref = torch.tensor(ref["sigma"], device="cuda", dtype=torch.float64)
    return mu_ref, sigma_ref
