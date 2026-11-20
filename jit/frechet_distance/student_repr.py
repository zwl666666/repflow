"""Frozen student representation model for FD loss."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("jit_fd")


class StudentReprModel(nn.Module):
    """Wrap distillation student (+ optional projector) as a frozen feature extractor.

    Input images are in [-1, 1] (same as JiT training). Returns one vector per image.
    """

    def __init__(
        self,
        net_student: nn.Module,
        projector: nn.Module | None = None,
        *,
        use_projector: bool = False,
        fd_t: float = 1.0,
        use_mask: bool = False,
        apply_final_norm: bool = True,
        pool: str = "mean",
    ):
        super().__init__()
        self.net_student = net_student
        self.projector = projector
        self.use_projector = use_projector and projector is not None
        self.fd_t = fd_t
        self.use_mask = use_mask
        self.apply_final_norm = apply_final_norm
        self.pool = pool

        self.net_student.eval().requires_grad_(False)
        if self.use_projector:
            self.projector.eval().requires_grad_(False)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, net_student.input_size, net_student.input_size, device="cuda")
            labels = torch.zeros(1, dtype=torch.long, device="cuda")
            feat = self.extract_features(dummy, labels)
        self.feat_dim = feat.shape[-1]
        logger.info(
            "[StudentReprModel] feat_dim=%d use_projector=%s fd_t=%.3f use_mask=%s apply_final_norm=%s pool=%s",
            self.feat_dim,
            self.use_projector,
            self.fd_t,
            self.use_mask,
            self.apply_final_norm,
            self.pool,
        )

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return tokens.mean(dim=1)
        if self.pool == "sum":
            return tokens.sum(dim=1)
        if self.pool == "flatten":
            return tokens.reshape(tokens.shape[0], -1)
        raise ValueError(f"Unknown pool mode: {self.pool}")

    def extract_features(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        t = torch.full((b,), self.fd_t, device=x.device, dtype=x.dtype)

        feat_kw = {}
        layers = getattr(self.net_student, "projector_feature_layers", None)
        if self.use_projector and layers is not None and len(layers) > 1:
            feat_kw["feature_layers"] = layers
        fea, _, keep_idx = self.net_student(x, t, labels, use_mask=self.use_mask, **feat_kw)
        if (
            self.apply_final_norm
            and not self.use_mask
            and hasattr(self.net_student, "norm")
            and fea.shape[-1] == self.net_student.hidden_size
        ):
            fea = self.net_student.norm(fea)

        if self.use_projector:
            if keep_idx is None:
                n = fea.shape[1]
                keep_idx = torch.arange(n, device=x.device).unsqueeze(0).expand(b, -1)
            proj_out = self.projector(fea, keep_idx, t)
            return self._pool_tokens(proj_out)

        return self._pool_tokens(fea)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x, labels)


def build_student_repr_model(
    denoiser: nn.Module,
    *,
    use_projector: bool = False,
    fd_t: float = 1.0,
    use_mask: bool = False,
    apply_final_norm: bool = True,
    pool: str = "mean",
) -> StudentReprModel:
    if not hasattr(denoiser, "net_student"):
        raise AttributeError("Denoiser has no net_student; load a distillation checkpoint.")
    return StudentReprModel(
        denoiser.net_student,
        getattr(denoiser, "projector", None),
        use_projector=use_projector,
        fd_t=fd_t,
        use_mask=use_mask,
        apply_final_norm=apply_final_norm,
        pool=pool,
    )
