"""Teacher-only flow-matching baseline (no student / projector distillation).

Optional REPA (DINOv2) alignment is gated by ``repa_loss_weight > 0``.
Optional SRA (self-representation alignment) is gated by ``sra_loss_weight > 0``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_distill_baseline import JiT_models
from repa_utils import (
    build_repa_mlp,
    dinov2_embed_dim,
    extract_dinov2_patch_tokens,
    extract_mae_patch_tokens,
    load_dinov2_encoder,
    load_mae_encoder,
    mae_embed_dim,
    preprocess_for_dinov2,
    preprocess_for_mae,
    repa_neg_cosine_loss,
)
from train_utils import (
    build_teacher_kwargs,
    merge_init_kwargs,
    shared_vision_kwargs,
    sra_alignment_loss,
    sra_layer_index,
)


class SimpleHead(nn.Module):
    """SRA alignment head (SiT-SRA): 2-layer SiLU MLP, hidden = in_dim + out_dim."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim + out_dim)
        self.linear2 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


def _teacher_init_kwargs(args) -> dict:
    cfg = merge_init_kwargs(shared_vision_kwargs(args), build_teacher_kwargs(args))
    # Baseline JiT has no DIFT / cosine branches.
    cfg.pop("teacher_feature_layers", None)
    cfg.pop("teacher_cons_layer", None)
    # Keep teacher_repa_layer when REPA is enabled (set by build_teacher_kwargs).
    return cfg


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = JiT_models[args.model](**_teacher_init_kwargs(args))
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.uncondition = getattr(args, "uncondition", False)

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema_decay = args.ema_decay
        self.ema_update_freq = max(1, int(getattr(args, "ema_update_freq", 1)))
        self.ema_params = None
        self._ema_step = 0

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        self.flow_loss_weight = getattr(args, "flow_loss_weight", 1.0)
        self.repa_loss_weight = float(getattr(args, "repa_loss_weight", 0.0) or 0.0)
        self.sra_loss_weight = float(getattr(args, "sra_loss_weight", 0.0) or 0.0)
        self.sra_t_max = float(getattr(args, "sra_t_max", 0.2) or 0.2)
        self.sra_loss_type = getattr(args, "sra_loss_type", "cos")
        self.sra_block_s = None
        self.sra_block_t = None
        self.ap_head = None
        if self.sra_loss_weight > 0:
            self.sra_block_s = sra_layer_index(getattr(args, "sra_block_out_s", 8))
            self.sra_block_t = sra_layer_index(getattr(args, "sra_block_out_t", 20))
            if self.sra_block_s >= self.sra_block_t:
                raise ValueError("SRA requires --sra_block_out_s < --sra_block_out_t (1-based).")
            self.ap_head = SimpleHead(self.net.hidden_size, self.net.hidden_size)

        # REPA (DINOv2 / MAE): default off; encoder kept out of state_dict / EMA.
        object.__setattr__(self, "repa_encoder", None)
        self.repa_projector = None
        self.repa_enc_type = getattr(args, "repa_enc_type", "dinov2-vit-b")
        self._repa_is_mae = "mae" in self.repa_enc_type
        if self.repa_loss_weight > 0:
            z_dim = (
                mae_embed_dim(self.repa_enc_type)
                if self._repa_is_mae
                else dinov2_embed_dim(self.repa_enc_type)
            )
            projector_dim = int(getattr(args, "repa_projector_dim", 2048))
            self.repa_projector = build_repa_mlp(
                self.net.hidden_size, projector_dim, z_dim
            )
            device = next(self.net.parameters()).device
            if self._repa_is_mae:
                encoder = load_mae_encoder(
                    self.repa_enc_type,
                    device=device,
                    resolution=int(args.img_size),
                    checkpoint_path=getattr(args, "repa_encoder_ckpt", None),
                )
            else:
                encoder = load_dinov2_encoder(
                    self.repa_enc_type,
                    device=device,
                    resolution=int(args.img_size),
                    repo_or_dir=getattr(args, "repa_encoder_repo", None),
                    checkpoint_path=getattr(args, "repa_encoder_ckpt", None),
                )
            object.__setattr__(self, "repa_encoder", encoder)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def _ema_net_param_dict(self):
        if self.ema_params is None:
            raise RuntimeError("SRA teacher forward requires ema_params.")
        ema_by_name = {
            name: ema_p
            for (name, _), ema_p in zip(self.named_parameters(), self.ema_params)
        }
        params = {}
        for name, param in self.net.named_parameters():
            key = f"net.{name}"
            if key not in ema_by_name:
                raise RuntimeError(f"SRA EMA is missing parameter {key}.")
            params[name] = ema_by_name[key].to(device=param.device, dtype=param.dtype)
        return params

    def _ema_net_layer_features(self, z, t_flat, labels, feature_layer):
        try:
            from torch.func import functional_call
        except ImportError:
            from torch.nn.utils.stateless import functional_call

        was_training = self.net.training
        self.net.eval()
        try:
            out = functional_call(
                self.net,
                self._ema_net_param_dict(),
                args=(z, t_flat, labels),
                kwargs={"compute_fea_repa": False, "feature_layer": feature_layer},
            )
        finally:
            self.net.train(was_training)
        if not isinstance(out, tuple) or len(out) < 2:
            raise RuntimeError("SRA EMA forward did not return layer features.")
        return out[1]

    def forward(self, x, labels, ema):
        del ema
        if self.uncondition:
            labels_dropped = torch.full_like(labels, self.num_classes)
        else:
            labels_dropped = self.drop_labels(labels) if self.training else labels
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        use_repa = self.repa_loss_weight > 0 and self.repa_projector is not None
        use_sra = (
            self.training
            and self.sra_loss_weight > 0
            and self.ap_head is not None
            and self.sra_block_s is not None
        )
        fea_repa = None
        fea_s = None
        if use_repa and use_sra:
            x_pred, fea_repa, fea_s = self.net(
                z,
                t.flatten(),
                labels_dropped,
                compute_fea_repa=True,
                feature_layer=self.sra_block_s,
            )
        elif use_repa:
            x_pred, fea_repa = self.net(
                z, t.flatten(), labels_dropped, compute_fea_repa=True
            )
        elif use_sra:
            x_pred, fea_s = self.net(
                z, t.flatten(), labels_dropped, feature_layer=self.sra_block_s
            )
        else:
            x_pred = self.net(z, t.flatten(), labels_dropped, compute_fea_repa=False)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        flow_loss = (v - v_pred).pow(2).mean(dim=(1, 2, 3)).mean()

        repa_loss = flow_loss.new_zeros(())
        if use_repa:
            if fea_repa is None:
                raise RuntimeError("REPA enabled but teacher fea_repa is None.")
            if self.repa_encoder is None:
                raise RuntimeError("REPA enabled but DINOv2 encoder is not loaded.")
            N, T, D = fea_repa.shape
            z_tilde = self.repa_projector(fea_repa.reshape(N * T, D)).reshape(N, T, -1)
            with torch.no_grad():
                enc = self.repa_encoder
                if next(enc.parameters()).device != x.device:
                    enc.to(x.device)
                if self._repa_is_mae:
                    enc_in = preprocess_for_mae(
                        x, resolution=self.img_size, input_range="neg1_1"
                    )
                    z_ref = extract_mae_patch_tokens(enc, enc_in)
                else:
                    enc_in = preprocess_for_dinov2(
                        x, resolution=self.img_size, input_range="neg1_1"
                    )
                    z_ref = extract_dinov2_patch_tokens(enc, enc_in)
            repa_loss = repa_neg_cosine_loss(z_ref, z_tilde)

        sra_loss = flow_loss.new_zeros(())
        if use_sra:
            if fea_s is None:
                raise RuntimeError("SRA enabled but online layer features are None.")
            # JiT: t=0 noise, t=1 data. Teacher is cleaner (larger t).
            t_teacher = (t + self.sra_t_max * torch.rand_like(t)).clamp(0, 1)
            z_teacher = t_teacher * x + (1 - t_teacher) * e
            with torch.no_grad():
                fea_t = self._ema_net_layer_features(
                    z_teacher, t_teacher.flatten(), labels_dropped, self.sra_block_t
                )
            xr = self.ap_head(fea_s)
            sra_loss = sra_alignment_loss(xr, fea_t, self.sra_loss_type)

        total_loss = (
            self.flow_loss_weight * flow_loss
            + self.repa_loss_weight * repa_loss
            + self.sra_loss_weight * sra_loss
        )
        aux_loss = self.repa_loss_weight * repa_loss + self.sra_loss_weight * sra_loss
        return total_loss, aux_loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(
            bsz, 3, self.img_size, self.img_size, device=device
        )
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(
            -1, *([1] * z.ndim)
        ).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(self.method)

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        x_cond = self.net(z, t.flatten(), labels, compute_fea_repa=False)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        if self.cfg_scale <= 1.0:
            return v_cond

        x_uncond = self.net(
            z,
            t.flatten(),
            torch.full_like(labels, self.num_classes),
            compute_fea_repa=False,
        )
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred

    def _ema_decay_for_update(self) -> float:
        if self.ema_update_freq <= 1:
            return self.ema_decay
        return self.ema_decay ** self.ema_update_freq

    @torch.no_grad()
    def maybe_update_ema(self, force: bool = False) -> bool:
        if self.ema_params is None:
            return False
        if not force:
            self._ema_step += 1
            if self._ema_step % self.ema_update_freq != 0:
                return False
        decay = self._ema_decay_for_update()
        blend = 1.0 - decay
        source_params = list(self.parameters())
        if source_params[0].device == self.ema_params[0].device:
            src_data = [p.detach() for p in source_params]
            torch._foreach_mul_(self.ema_params, decay)
            torch._foreach_add_(self.ema_params, src_data, alpha=blend)
        else:
            for targ, src in zip(self.ema_params, source_params):
                src_data = src.detach().to(device=targ.device, dtype=targ.dtype)
                targ.mul_(decay).add_(src_data, alpha=blend)
        return True

    @torch.no_grad()
    def update_ema(self):
        self.maybe_update_ema(force=True)
