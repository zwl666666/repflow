#!/usr/bin/env python3
"""Sweep teacher linear probe: layer x flow-noise timestep (uncond nodual SiT teacher).

Uses frozen SiT-XL/2 EMA + VAE latents. See JiT/linear_teacher_sweep.py for the same
coarse/fine sweep strategy and acceleration notes.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models import AutoencoderKL
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from linear_student import build_dataset, build_loader
from models import SiT_models
from train_utils import center_crop_arr


def print0(msg: str, local_rank: int) -> None:
    if local_rank in (-1, 0):
        print(msg, flush=True)


def init_dist(local_rank: int) -> Tuple[torch.device, int]:
    # torchrun 只设置 LOCAL_RANK 环境变量，不一定再传 --local-rank
    if local_rank < 0:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0 and torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"local_rank {local_rank} >= available gpus {torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return device, local_rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, -1


def avg_pool_tokens(act: torch.Tensor) -> torch.Tensor:
    if act.dim() == 3:
        return act.float().mean(dim=1)
    act = act.view(act.shape[0], act.shape[1], -1).float()
    return act.mean(dim=1)


def flow_perturb_latent(latent: torch.Tensor, timestep: int) -> torch.Tensor:
    t = float(timestep) / 1000.0
    noise = torch.randn_like(latent)
    return t * latent + (1.0 - t) * noise


def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def default_coarse_layers(depth: int) -> List[int]:
    layers = list(range(0, depth, 4))
    if (depth - 1) not in layers:
        layers.append(depth - 1)
    return layers


def default_coarse_timesteps() -> List[int]:
    return [100, 250, 400, 550, 700, 850]


def combo_key(layer: int, timestep: int) -> str:
    return f"layer-{layer}/t-{timestep}"


@dataclass(frozen=True)
class Combo:
    layer: int
    timestep: int


def build_fine_combos(depth: int, coarse_best: dict) -> List[Combo]:
    bl, bt = int(coarse_best["layer"]), int(coarse_best["timestep"])
    layers = sorted({max(0, min(depth - 1, bl + d)) for d in (-4, -2, 0, 2, 4)})
    timesteps = sorted({max(1, min(999, bt + d)) for d in (-150, -75, 0, 75, 150)})
    return [Combo(layer=l, timestep=t) for l in layers for t in timesteps]


def build_combos_for_phase(
    phase: str,
    depth: int,
    layers_spec: str,
    timesteps_spec: str,
    coarse_best: Optional[dict] = None,
    coarse_result_path: str = "",
) -> List[Combo]:
    if phase == "coarse":
        layers = default_coarse_layers(depth)
        timesteps = default_coarse_timesteps()
    elif phase == "fine":
        best = coarse_best
        if best is None:
            if not coarse_result_path or not os.path.isfile(coarse_result_path):
                raise FileNotFoundError("fine phase needs coarse_best or --coarse-result JSON")
            best = json.loads(Path(coarse_result_path).read_text())["best"]
        return build_fine_combos(depth, best)
    else:
        layers = parse_int_list(layers_spec)
        timesteps = parse_int_list(timesteps_spec)
    return [Combo(layer=l, timestep=t) for l in layers for t in timesteps]


def load_teacher_model(ckpt_path: str, model_name: str, device: torch.device, use_ema: bool) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args")
    image_size = getattr(ckpt_args, "image_size", 256) if ckpt_args is not None else 256
    num_classes = getattr(ckpt_args, "num_classes", 1000) if ckpt_args is not None else 1000
    latent_channels = getattr(ckpt_args, "latent_channels", 4) if ckpt_args is not None else 4
    model = SiT_models[model_name](input_size=image_size // 8, num_classes=num_classes, in_channels=latent_channels)
    if use_ema and ckpt.get("ema") is not None:
        state = ckpt["ema"]
        source = "ema"
    else:
        state = ckpt["model"]
        source = "model"
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print0(f"[warn] missing teacher keys: {len(missing)}", 0)
        print0(f"  missing: {missing}", 0)
    if unexpected:
        print0(f"[warn] unexpected teacher keys: {len(unexpected)}", 0)
        print0(f"  unexpected: {unexpected}", 0)
    print0(f"Loaded teacher from {source}: {ckpt_path}", 0)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class SiTTeacherFeatureExtractor:
    def __init__(
        self,
        model: nn.Module,
        vae: AutoencoderKL,
        device: torch.device,
        null_class: int,
        use_amp: bool,
    ):
        self.model = model
        self.vae = vae
        self.device = device
        self.null_class = null_class
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16

    @torch.no_grad()
    def extract_all_layers(self, images: torch.Tensor, timestep: int, layer_ids: Sequence[int]) -> Dict[int, torch.Tensor]:
        # no_grad (not inference_mode): features must be usable by probe BatchNorm backward.
        images = images.to(self.device, non_blocking=True).float().div_(255.0).mul_(2.0).sub_(1.0)
        autocast_ctx = torch.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
        with autocast_ctx:
            latent = self.vae.encode(images).latent_dist.sample().mul_(0.18215)
        x = flow_perturb_latent(latent.float(), timestep)
        batch = x.size(0)
        y = torch.full((batch,), self.null_class, device=self.device, dtype=torch.long)
        t = torch.zeros(batch, device=self.device)

        captured: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(idx: int):
            def _hook(_module, _inp, out):
                captured[idx] = avg_pool_tokens(out)

            return _hook

        for idx in layer_ids:
            hooks.append(self.model.blocks[idx].register_forward_hook(make_hook(idx)))

        with autocast_ctx:
            _ = self.model(x, t, y)

        for h in hooks:
            h.remove()
        return {k: captured[k].float() for k in layer_ids}


class MultiComboProbe(nn.Module):
    def __init__(
        self,
        combos: Sequence[Combo],
        feat_dim: int,
        num_classes: int,
        base_lr: float,
        total_epochs: int,
        device: torch.device,
        local_rank: int,
    ):
        super().__init__()
        self.combos = list(combos)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
        self.local_rank = local_rank
        heads = nn.ModuleDict(
            {
                combo_key(c.layer, c.timestep): nn.Sequential(
                    nn.BatchNorm1d(feat_dim), nn.Linear(feat_dim, num_classes)
                )
                for c in combos
            }
        )
        for head in heads.values():
            nn.init.normal_(head[1].weight, std=0.01)
            nn.init.zeros_(head[1].bias)
        heads = heads.to(device)
        if dist.is_available() and dist.is_initialized() and device.type == "cuda":
            gpu = device.index if device.index is not None else torch.cuda.current_device()
            self.heads = nn.parallel.DistributedDataParallel(heads, device_ids=[gpu], output_device=gpu)
        else:
            self.heads = heads
        module = self.heads.module if hasattr(self.heads, "module") else self.heads
        self.optims = {key: torch.optim.Adam(module[key].parameters(), lr=base_lr) for key in module.keys()}
        self.schedulers = {key: CosineAnnealingLR(opt, T_max=max(1, total_epochs)) for key, opt in self.optims.items()}

    def _head(self, layer: int, timestep: int) -> nn.Module:
        module = self.heads.module if hasattr(self.heads, "module") else self.heads
        return module[combo_key(layer, timestep)]

    def train_step(self, feats_by_layer: Dict[int, torch.Tensor], labels: torch.Tensor, timestep: int, layer_ids: Sequence[int]):
        labels = labels.to(self.device, non_blocking=True)
        for layer in layer_ids:
            key = combo_key(layer, timestep)
            feats = feats_by_layer[layer].detach().clone()
            logits = self._head(layer, timestep)(feats)
            loss = self.loss_fn(logits, labels)
            self.optims[key].zero_grad(set_to_none=True)
            loss.backward()
            self.optims[key].step()

    @torch.inference_mode()
    def predict(self, feats_by_layer: Dict[int, torch.Tensor], timestep: int, layer: int) -> torch.Tensor:
        return self._head(layer, timestep)(feats_by_layer[layer]).argmax(dim=-1)

    def schedule_step(self):
        for scheduler in self.schedulers.values():
            scheduler.step()


@torch.inference_mode()
def eval_combo(
    probe: MultiComboProbe,
    extractor: SiTTeacherFeatureExtractor,
    val_loader: DataLoader,
    combos: Sequence[Combo],
    max_batches: Optional[int],
    local_rank: int,
) -> Dict[str, float]:
    layer_ids = sorted({c.layer for c in combos})
    stats = {combo_key(c.layer, c.timestep): {"correct": 0, "total": 0} for c in combos}
    for bi, (images, labels) in enumerate(tqdm(val_loader, disable=(local_rank not in (-1, 0)))):
        if max_batches is not None and bi >= max_batches:
            break
        labels = labels.to(probe.device, non_blocking=True)
        for t in sorted({c.timestep for c in combos}):
            feats = extractor.extract_all_layers(images, t, layer_ids)
            for c in combos:
                if c.timestep != t:
                    continue
                key = combo_key(c.layer, c.timestep)
                pred = probe.predict(feats, t, c.layer)
                stats[key]["correct"] += (pred == labels).sum().item()
                stats[key]["total"] += labels.numel()

    out = {}
    for key, st in stats.items():
        correct = torch.tensor(st["correct"], device=probe.device, dtype=torch.long)
        total = torch.tensor(st["total"], device=probe.device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        out[key] = (correct.float() / total.float()).item() if total.item() else 0.0
    return out


def infer_feat_dim(extractor: SiTTeacherFeatureExtractor, image_size: int, layer: int, timestep: int) -> int:
    dummy = torch.zeros(1, 3, image_size, image_size)
    return extractor.extract_all_layers(dummy, timestep, [layer])[layer].shape[1]


def run_sweep_phase(
    phase: str,
    combos: Sequence[Combo],
    extractor: SiTTeacherFeatureExtractor,
    train_loader: DataLoader,
    train_sampler,
    val_loader: DataLoader,
    args,
    device: torch.device,
    local_rank: int,
    world: int,
) -> dict:
    layer_ids = sorted({c.layer for c in combos})
    timesteps = sorted({c.timestep for c in combos})
    epochs = args.probe_epochs_coarse if phase == "coarse" else args.probe_epochs
    max_val_batches = args.max_val_batches
    if phase == "coarse" and max_val_batches is None:
        max_val_batches = 50

    print0(f"SiT teacher sweep [{phase}]: combos={len(combos)}, epochs={epochs}", local_rank)
    print0(f"  layers={layer_ids}", local_rank)
    print0(f"  timesteps={timesteps}", local_rank)
    max_train_batches = getattr(args, "max_train_batches", None)
    if max_train_batches is not None:
        print0(f"  max_train_batches={max_train_batches}", local_rank)

    feat_dim = infer_feat_dim(extractor, args.image_size, layer_ids[0], timesteps[0])
    probe = MultiComboProbe(
        combos, feat_dim, args.num_classes, args.lr * world, epochs, device, local_rank
    ).to(device)

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for bi, (images, labels) in enumerate(tqdm(train_loader, disable=(local_rank not in (-1, 0)))):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            for t in timesteps:
                feats = extractor.extract_all_layers(images, t, layer_ids)
                probe.train_step(feats, labels, t, layer_ids)
        probe.schedule_step()

    accs = eval_combo(probe, extractor, val_loader, combos, max_val_batches, local_rank)
    results = [
        {"layer": c.layer, "timestep": c.timestep, "val_acc": accs[combo_key(c.layer, c.timestep)]}
        for c in combos
    ]
    best = max(results, key=lambda r: r["val_acc"])
    print0(
        f"[{phase}] best: layer={best['layer']} t={best['timestep']} acc={best['val_acc']:.4f}",
        local_rank,
    )
    return {
        "phase": phase,
        "probe_epochs": epochs,
        "best": best,
        "results": sorted(results, key=lambda r: (-r["val_acc"], r["layer"], r["timestep"])),
    }


def main():
    parser = argparse.ArgumentParser(description="SiT teacher linear probe layer x noise sweep")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="SiT-XL/2")
    parser.add_argument("--vae", default="./sd-vae-ft-ema")
    parser.add_argument("--use-teacher-ema", action="store_true", default=True)
    parser.add_argument("--no-use-teacher-ema", dest="use_teacher_ema", action="store_false")
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--val-data-path", required=True)
    parser.add_argument("--train-label-file", required=True)
    parser.add_argument("--val-label-file", required=True)
    parser.add_argument("--petrel-conf", default=os.environ.get("PETRELOSS_CONF"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--null-class", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-epochs-coarse", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument(
        "--sweep-mode",
        choices=["auto", "coarse", "fine", "custom"],
        default="auto",
        help="auto=coarse then fine in one run (default)",
    )
    parser.add_argument("--layers", default="")
    parser.add_argument("--timesteps", default="")
    parser.add_argument("--coarse-result", default="")
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Cap train batches per epoch (for fast search).",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=-1)
    args, _ = parser.parse_known_args()

    device, local_rank = init_dist(args.local_rank)
    world = dist.get_world_size() if dist.is_initialized() else 1
    per_gpu_batch = max(1, args.global_batch_size // world)
    print0(
        f"dist: rank={local_rank} world={world} device={device} per_gpu_batch={per_gpu_batch}",
        local_rank,
    )

    model = load_teacher_model(args.ckpt, args.model, device, args.use_teacher_ema)
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    depth = len(model.blocks)
    phases = ["coarse", "fine"] if args.sweep_mode == "auto" else [args.sweep_mode]

    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.PILToTensor(),
    ])
    train_set = build_dataset(args.train_data_path, args.train_label_file, transform_train, petrel_conf=args.petrel_conf)
    val_set = build_dataset(args.val_data_path, args.val_label_file, transform_val, petrel_conf=args.petrel_conf)
    train_loader, train_sampler = build_loader(train_set, per_gpu_batch, True, args.num_workers, True)
    val_loader, _ = build_loader(val_set, per_gpu_batch, False, args.num_workers, True)

    extractor = SiTTeacherFeatureExtractor(model, vae, device, args.null_class, args.use_amp)
    phase_payloads = {}
    coarse_best = None
    for phase in phases:
        combos = build_combos_for_phase(
            phase, depth, args.layers, args.timesteps, coarse_best, args.coarse_result
        )
        phase_payloads[phase] = run_sweep_phase(
            phase,
            combos,
            extractor,
            train_loader,
            train_sampler,
            val_loader,
            args,
            device,
            local_rank,
            world,
        )
        if phase == "coarse":
            coarse_best = phase_payloads[phase]["best"]

    if args.sweep_mode == "auto":
        payload = {
            "framework": "SiT",
            "model": args.model,
            "ckpt": args.ckpt,
            "sweep_mode": "auto",
            "coarse": phase_payloads["coarse"],
            "fine": phase_payloads["fine"],
            "best": phase_payloads["fine"]["best"],
        }
    else:
        single = phase_payloads[phases[0]]
        payload = {
            "framework": "SiT",
            "model": args.model,
            "ckpt": args.ckpt,
            "sweep_mode": args.sweep_mode,
            "probe_epochs": single["probe_epochs"],
            "best": single["best"],
            "results": single["results"],
        }

    if local_rank in (-1, 0):
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print0(f"Saved sweep results -> {out_path}", local_rank)
        best = payload["best"]
        print0(f"Overall best: layer={best['layer']} t={best['timestep']} acc={best['val_acc']:.4f}", local_rank)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
