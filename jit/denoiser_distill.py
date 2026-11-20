import torch
import torch.nn as nn
from model_distill import JiT_models, JiT_models_student, Projector_models
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F

from repa_utils import (
    build_repa_mlp,
    dinov2_embed_dim,
    extract_dinov2_patch_tokens,
    load_dinov2_encoder,
    preprocess_for_dinov2,
    repa_neg_cosine_loss,
)
from train_utils import (
    build_projector_kwargs,
    build_student_kwargs,
    build_teacher_kwargs,
    merge_init_kwargs,
    projector_feature_layers_list,
    shared_vision_kwargs,
)


def _projector_kwargs(args, student_hidden_size: int) -> dict:
    cfg = merge_init_kwargs(shared_vision_kwargs(args), build_projector_kwargs(args))
    cfg.setdefault("input_hidden_size", student_hidden_size)
    if "attn_drop" not in cfg:
        cfg["attn_drop"] = (
            getattr(args, "projector_attn_dropout", None)
            if getattr(args, "projector_attn_dropout", None) is not None
            else args.attn_dropout
        )
    if "proj_drop" not in cfg:
        cfg["proj_drop"] = (
            getattr(args, "projector_proj_dropout", None)
            if getattr(args, "projector_proj_dropout", None) is not None
            else args.proj_dropout
        )
    return cfg


class TeacherConsAlign(nn.Module):
    """将 teacher consistency 特征投影到 student hidden 维度。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.linear(x)


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        teacher_cfg = merge_init_kwargs(shared_vision_kwargs(args), build_teacher_kwargs(args))
        student_cfg = merge_init_kwargs(shared_vision_kwargs(args), build_student_kwargs(args))

        self.net = JiT_models[args.model](**teacher_cfg)
        self.net_student = JiT_models_student[args.student_model](**student_cfg)
        self.projector_feature_layers = projector_feature_layers_list(
            args, len(self.net_student.blocks)
        )
        self.net_student.projector_feature_layers = self.projector_feature_layers
        if getattr(args, "skip_projector", False):
            self.projector = None
        else:
            projector_model_key = getattr(args, "projector_model", None) or args.student_model
            projector_in_dim = self.net_student.hidden_size * len(self.projector_feature_layers)
            self.projector = Projector_models[projector_model_key](
                **_projector_kwargs(args, projector_in_dim)
            )
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.uncondition = getattr(args, "uncondition", False)

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay = args.ema_decay
        self.ema_update_freq = max(1, int(getattr(args, "ema_update_freq", 1)))
        self.ema_params = None
        self._ema_step = 0

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        self.flow_loss_weight = getattr(args, "flow_loss_weight", 1.0)
        self.distill_loss_weight = getattr(args, "distill_loss_weight", 1.0)
        self.distill_mse_weight = getattr(args, "distill_mse_weight", 1.0)
        self.distill_cosine_weight = getattr(args, "distill_cosine_weight", 0.1)
        self.repa_loss_weight = float(getattr(args, "repa_loss_weight", 0.0) or 0.0)

        # teacher ProjLayer 固定输出 1024 维；仅 cos loss>0 时对齐到 student
        self.teacher_cons_dim = 1024
        if (
            self.distill_cosine_weight > 0
            and self.net_student.hidden_size != self.teacher_cons_dim
        ):
            self.teacher_cons_proj = TeacherConsAlign(
                self.teacher_cons_dim, self.net_student.hidden_size
            )
        else:
            self.teacher_cons_proj = None

        # REPA (DINOv2): default off; only build when weight > 0
        # Keep encoder OUT of nn.Module state_dict / EMA / DDP params (frozen teacher).
        object.__setattr__(self, "repa_encoder", None)
        self.repa_projector = None
        self.repa_enc_type = getattr(args, "repa_enc_type", "dinov2-vit-b")
        if self.repa_loss_weight > 0:
            z_dim = dinov2_embed_dim(self.repa_enc_type)
            projector_dim = int(getattr(args, "repa_projector_dim", 2048))
            self.repa_projector = build_repa_mlp(
                self.net.hidden_size, projector_dim, z_dim
            )
            device = next(self.net.parameters()).device
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
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def masked_mse_loss(self, fea, fea_cons, mask, t=None, eps=1e-6):
        if mask is None:
            return fea.new_zeros(())
        assert fea.shape == fea_cons.shape
        assert mask.shape == fea.shape[:2]
        loss = F.smooth_l1_loss(fea, fea_cons, reduction="none").mean(dim=-1)
        mask_float = mask.float()
        return (loss * mask_float).sum() / (mask_float.sum() + eps)

    def cosine_loss(self, fea, fea_cons, batch=False, eps=1e-6):
        if batch:
            B = fea.shape[0]
            fea = fea.reshape(B, -1)
            fea_cons = fea_cons.reshape(B, -1)
        fea_n = F.normalize(fea, dim=-1, eps=eps)
        fea_cons_n = F.normalize(fea_cons, dim=-1, eps=eps)
        cos = (fea_n * fea_cons_n).sum(dim=-1)
        loss = 1.0 - cos
        return loss.mean()

    def forward(self, x, labels, ema):
        if self.uncondition:
            labels_dropped = torch.full_like(labels, self.num_classes)
        else:
            labels_dropped = self.drop_labels(labels) if self.training else labels
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        use_cosine = self.distill_cosine_weight > 0
        use_repa = self.repa_loss_weight > 0 and self.repa_projector is not None
        x_pred, fea_pred, fea_cons, fea_repa = self.net(
            z,
            t.flatten(),
            labels_dropped,
            compute_fea_cons=use_cosine,
            compute_fea_repa=use_repa,
        )
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        student_kw = {}
        if len(self.projector_feature_layers) > 1:
            student_kw["feature_layers"] = self.projector_feature_layers
        fea, mask, keep_idx = self.net_student(
            x, t.flatten(), labels_dropped, **student_kw
        )
        fea = self.projector(fea, keep_idx, t)

        with torch.no_grad():
            fea_pred = F.layer_norm(fea_pred, (fea_pred.size(-1),))
            if use_cosine:
                ema.eval()
                fea_anchor, _, _ = ema.net_student(
                    x, t.flatten(), labels, use_mask=False
                )

        cons_mse = self.masked_mse_loss(fea, fea_pred.detach(), mask)
        if use_cosine:
            if self.teacher_cons_proj is not None:
                fea_cons_for_cos = self.teacher_cons_proj(fea_cons)
            else:
                fea_cons_for_cos = fea_cons
            cons_cos = self.cosine_loss(fea_anchor.detach(), fea_cons_for_cos)
            cons_loss = (
                self.distill_mse_weight * cons_mse
                + self.distill_cosine_weight * cons_cos
            )
        else:
            cons_loss = self.distill_mse_weight * cons_mse

        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        repa_loss = loss.new_zeros(())
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
                dino_in = preprocess_for_dinov2(
                    x, resolution=self.img_size, input_range="neg1_1"
                )
                z_dino = extract_dinov2_patch_tokens(enc, dino_in)
            repa_loss = repa_neg_cosine_loss(z_dino, z_tilde)

        total_loss = (
            self.flow_loss_weight * loss
            + self.distill_loss_weight * cons_loss
            + self.repa_loss_weight * repa_loss
        )
        # Fold REPA into the logged aux loss so engine stays backward-compatible.
        return total_loss, cons_loss + self.repa_loss_weight * repa_loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(
            -1, *([1] * z.ndim)
        ).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    def _forward_sample_impl(self, z, t, labels, cfg_scale=None, cfg_interval=None):
        if cfg_scale is None:
            cfg_scale = self.cfg_scale
        if cfg_interval is None:
            cfg_interval = self.cfg_interval

        x_cond, _, _, _ = self.net(
            z,
            t.flatten(),
            labels,
            compute_features=False,
            compute_fea_cons=False,
            compute_fea_repa=False,
        )
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        if cfg_scale <= 1.0:
            return v_cond

        x_uncond, _, _, _ = self.net(
            z,
            t.flatten(),
            torch.full_like(labels, self.num_classes),
            compute_features=False,
            compute_fea_cons=False,
            compute_fea_repa=False,
        )
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        low, high = cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, cfg_scale, 1.0)
        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        return self._forward_sample_impl(z, t, labels)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    def sample_images_with_grad(
        self,
        labels,
        num_steps=None,
        cfg_scale=None,
        cfg_interval=None,
    ):
        """Differentiable sampling for FD post-training (t: 0 noise -> 1 data)."""
        device = labels.device
        bsz = labels.size(0)
        num_steps = self.steps if num_steps is None else num_steps
        cfg_scale = 1.0 if cfg_scale is None else cfg_scale
        if cfg_interval is None:
            cfg_interval = self.cfg_interval

        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = timesteps[i].expand(bsz).view(-1, 1, 1, 1)
            t_next = timesteps[i + 1].expand(bsz).view(-1, 1, 1, 1)
            v_pred = self._forward_sample_impl(z, t, labels, cfg_scale, cfg_interval)
            z = z + (t_next - t) * v_pred
        return z

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
