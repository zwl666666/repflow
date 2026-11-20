import logging

import torch

logger = logging.getLogger("jit_fd")


class FeatureQueue(torch.nn.Module):
    """Circular buffer of features for FD computation."""

    def __init__(
        self,
        size: int = 50000,
        feat_dim: int = 2048,
        online_accum: bool = False,
        ema_beta: float = 0.0,
    ):
        super().__init__()
        self.size = size
        self.feat_dim = feat_dim
        self.online_accum = online_accum
        self.ema_beta = ema_beta
        self.ema_stats = ema_beta > 0.0

        if self.ema_stats:
            self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
            self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
            self.register_buffer("_ema_count", torch.zeros(1, dtype=torch.long))
        else:
            self.register_buffer("feats", torch.empty(size, feat_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            if online_accum and size > 0:
                self.register_buffer("feat_sum_old", torch.zeros(feat_dim, dtype=torch.float64))
                self.register_buffer("feat_outer_old", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))

    @property
    def pointer(self) -> int:
        return int(self.ptr.item())

    @torch.no_grad()
    def _init_accumulators(self):
        feats_d = self.feats.double()
        self.feat_sum_old.copy_(feats_d.sum(0))
        self.feat_outer_old.copy_(feats_d.T @ feats_d)

    @torch.no_grad()
    def accumulate_batch(self, feats: torch.Tensor):
        feats_d = feats.detach().float().double()
        self.mu_ema.add_(feats_d.sum(0))
        self.m2_ema.addmm_(feats_d.T, feats_d)
        self._ema_count += feats_d.shape[0]

    @torch.no_grad()
    def _finalize_streaming_init(self):
        count = self._ema_count.item()
        if count == 0:
            logger.warning("[FeatureQueue] EMA streaming init: no features accumulated")
            return
        self.mu_ema.div_(count)
        self.m2_ema.div_(count)
        logger.info("[FeatureQueue] EMA init done: %d features (beta=%s)", count, self.ema_beta)

    def build_feats_stats(self, new_feats: torch.Tensor):
        if self.ema_stats:
            return self._build_feats_stats_ema(new_feats)

        new_d = new_feats.double()
        b = new_d.shape[0]
        n = self.size

        evicted = self._get_evicted_feats(b).double()
        sum_old = self.feat_sum_old - evicted.sum(0)
        outer_old = self.feat_outer_old - evicted.T @ evicted

        feat_sum = sum_old.detach() + new_d.sum(0)
        feat_outer = outer_old.detach() + new_d.T @ new_d

        mu = feat_sum / n
        sigma = (feat_outer - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / n) / (n - 1)
        return mu, sigma

    def _build_feats_stats_ema(self, new_feats: torch.Tensor):
        beta = self.ema_beta
        new_d = new_feats.double()
        b = new_d.shape[0]

        mu = beta * self.mu_ema.detach() + (1.0 - beta) * new_d.mean(0)
        m2 = beta * self.m2_ema.detach() + (1.0 - beta) * (new_d.T @ new_d) / b
        sigma = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
        return mu, sigma

    def _snapshot(self, buf: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return new
        n = new.shape[0]
        snap = buf.clone().detach()
        ptr = self.pointer
        if ptr + n <= self.size:
            snap[ptr : ptr + n] = new
        else:
            first = self.size - ptr
            snap[ptr : self.size] = new[:first]
            snap[: n - first] = new[first:]
        return snap

    def build_feats_snapshot(self, new_feats: torch.Tensor) -> torch.Tensor:
        return self._snapshot(self.feats, new_feats)

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor):
        if self.size == 0:
            return

        n = new_feats.shape[0]
        new_det = new_feats.detach().float()

        if self.ema_stats:
            beta = self.ema_beta
            new_d = new_det.double()
            self.mu_ema.mul_(beta).add_(new_d.mean(0), alpha=1.0 - beta)
            self.m2_ema.mul_(beta).addmm_(new_d.T, new_d, alpha=(1.0 - beta) / n)
            return

        ptr = self.pointer

        if self.online_accum:
            evicted = self._get_evicted_feats(n).double()
            new_d = new_det.double()
            self.feat_sum_old.add_(new_d.sum(0) - evicted.sum(0))
            self.feat_outer_old.add_(new_d.T @ new_d - evicted.T @ evicted)

        if ptr + n <= self.size:
            self.feats[ptr : ptr + n] = new_det
        else:
            first = self.size - ptr
            self.feats[ptr : self.size] = new_det[:first]
            self.feats[: n - first] = new_det[first:]
        self.ptr[0] = (ptr + n) % self.size

    def _get_evicted_feats(self, n: int) -> torch.Tensor:
        ptr = self.pointer
        if ptr + n <= self.size:
            return self.feats[ptr : ptr + n]
        first = self.size - ptr
        return torch.cat([self.feats[ptr : self.size], self.feats[: n - first]], dim=0)
