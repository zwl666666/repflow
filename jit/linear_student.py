"""Linear probe on frozen JiT student features (SiT-main/linear_student.py style)."""

from __future__ import annotations

import os
import PIL.Image as Image
from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from util.crop import center_crop_arr


def print0(message: str, local_rank: int) -> None:
    if local_rank in (-1, 0):
        print(message, flush=True)


def as_torch_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def _avg_pool_patches(act: torch.Tensor) -> torch.Tensor:
    if act.dim() == 4:
        act = act.view(act.shape[0], act.shape[1], -1).float()
        return act.mean(dim=2)
    if act.dim() == 3:
        act = act.float()
        if act.shape[1] < act.shape[2]:
            return act.mean(dim=1)
        return act.mean(dim=2)
    act = act.view(act.shape[0], act.shape[1], -1).float()
    return act.mean(dim=2)


class LocalLabelFileDataset(torch.utils.data.Dataset):
    """Image dataset driven by a text label file ("<relative/path> <label>" per line)."""

    def __init__(self, root: str, label_file: str, transform: Optional[Callable] = None):
        self.root = root.rstrip("/")
        self.transform = transform
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Local data root not found: {self.root}")
        if not os.path.isfile(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        self.samples: List[Tuple[str, int]] = []
        with open(label_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.rsplit(maxsplit=1)
                self.samples.append((rel_path, int(label)))
        if not self.samples:
            raise RuntimeError(f"No samples found in label file: {label_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image_path = os.path.join(self.root, rel_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_dataset(root: str, label_file: Optional[str], transform):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Local data root not found: {root}")
    if label_file is not None:
        return LocalLabelFileDataset(root=root, label_file=label_file, transform=transform)
    from torchvision.datasets import ImageFolder
    return ImageFolder(root, transform=transform)

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Local data root not found: {root}")
    if label_file is not None:
        return LocalLabelFileDataset(root=root, label_file=label_file, transform=transform)
    from torchvision.datasets import ImageFolder

    return ImageFolder(root, transform=transform)


def build_loader(dataset, batch_size: int, train: bool, num_workers: int, pin_memory: bool):
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=train, drop_last=train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and train),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=train,
    )
    return loader, sampler


class JiTStudentFeatureExtractor:
    def __init__(
        self,
        student: nn.Module,
        device: torch.device,
        use_amp: bool,
        image_size: int,
        probe_timestep: float = 0.0,
        null_class: int = 1000,
    ):
        self.student = student
        self.device = device
        self.image_size = image_size
        self.probe_timestep = probe_timestep
        self.null_class = null_class
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        images = images.float().div_(255.0).mul_(2.0).sub_(1.0)
        batch_size = images.size(0)
        t = torch.full((batch_size,), self.probe_timestep, device=self.device, dtype=torch.float32)
        y = torch.full((batch_size,), self.null_class, device=self.device, dtype=torch.long)

        autocast_ctx = torch.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
        with autocast_ctx:
            tokens, _, _ = self.student(images, t, y, use_mask=False)
            if hasattr(self.student, "norm"):
                tokens = self.student.norm(tokens)
        return _avg_pool_patches(tokens).float()


def _average_grads(params) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    for param in params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        feature_extractor: JiTStudentFeatureExtractor,
        feat_dim: int,
        num_classes: int,
        base_lr: float,
        total_epochs: int,
        local_rank: int,
        device: torch.device,
        use_ddp: bool = True,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.device = device
        self.use_ddp = use_ddp
        self.loss_fn = nn.CrossEntropyLoss()

        head = nn.Sequential(
            nn.BatchNorm1d(feat_dim, affine=False),
            nn.Linear(feat_dim, num_classes, bias=True),
        ).to(device)
        nn.init.normal_(head[1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(head[1].bias)

        if use_ddp and dist.is_available() and dist.is_initialized():
            gpu_id = torch.cuda.current_device()
            self.classifier = torch.nn.parallel.DistributedDataParallel(
                head,
                device_ids=[gpu_id],
                output_device=gpu_id,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.classifier = head

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=base_lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_epochs))

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def schedule_step(self) -> None:
        self.scheduler.step()

    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        self.classifier.train()
        labels = labels.to(self.device, non_blocking=True)
        with torch.inference_mode():
            feats = self.feature_extractor(images)
        logits = self.classifier(feats)
        loss = self.loss_fn(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not self.use_ddp:
            _average_grads(self.classifier.parameters())
        self.optimizer.step()
        return loss.item()

    @torch.inference_mode()
    def test_step(self, images: torch.Tensor) -> torch.Tensor:
        self.classifier.eval()
        feats = self.feature_extractor(images)
        logits = self.classifier(feats)
        return logits.argmax(dim=-1)


@torch.no_grad()
def test_acc(classifier: LinearClassifier, val_loader, device, local_rank: int) -> float:
    correct = 0
    total = 0
    for images, labels in tqdm(val_loader, disable=(local_rank not in (-1, 0))):
        pred = classifier.test_step(images)
        labels = labels.to(device, non_blocking=True)
        correct += (pred == labels).sum().item()
        total += labels.numel()

    correct_t = torch.tensor(correct, device=device, dtype=torch.long)
    total_t = torch.tensor(total, device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
    return (correct_t.float() / total_t.float()).item()


def infer_feat_dim(feature_extractor: JiTStudentFeatureExtractor, image_size: int) -> int:
    dummy = torch.zeros(1, 3, image_size, image_size, device=feature_extractor.device)
    return feature_extractor(dummy).shape[1]


def _prepare_student_for_probe(student):
    """Unwrap DDP and freeze student for feature extraction (SiT linear_student.py style)."""
    student = student.module if hasattr(student, "module") else student
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)
    return student


def run_linear_probe(args, device, local_rank: int, student=None, use_ddp: bool = True):
    """Train a linear classifier on frozen JiT student features."""
    device = as_torch_device(device)
    if student is None:
        raise ValueError("run_linear_probe requires an in-process JiT student module.")

    student = _prepare_student_for_probe(student)

    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.PILToTensor(),
    ])

    train_set = build_dataset(
        args.train_data_path,
        args.train_label_file,
        transform_train,
    )
    val_set = build_dataset(
        args.val_data_path,
        args.val_label_file,
        transform_val,
    )
    train_loader, train_sampler = build_loader(
        train_set,
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    val_loader, _ = build_loader(
        val_set,
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    feature_extractor = JiTStudentFeatureExtractor(
        student=student,
        device=device,
        use_amp=args.use_amp,
        image_size=args.image_size,
        null_class=args.num_classes,
    )
    feat_dim = infer_feat_dim(feature_extractor, args.image_size)
    classifier = LinearClassifier(
        feature_extractor=feature_extractor,
        feat_dim=feat_dim,
        num_classes=args.num_classes,
        base_lr=args.lr,
        total_epochs=args.epochs,
        local_rank=local_rank,
        device=device,
        use_ddp=use_ddp,
    ).to(device)

    best_acc = 0.0
    epoch_accs = []
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, disable=(local_rank not in (-1, 0)))
        for step, (images, labels) in enumerate(pbar):
            loss = classifier.train_step(images, labels)
            if local_rank in (-1, 0):
                pbar.set_description(
                    f"[linear epoch {epoch} / iter {step}] lr={classifier.get_lr():.2e} loss={loss:.4f}"
                )

        classifier.schedule_step()
        acc = test_acc(classifier, val_loader, device=device, local_rank=local_rank)
        epoch_accs.append(acc)
        best_acc = max(best_acc, acc)
        print0(f"[linear epoch {epoch}] acc: {acc * 100:.2f}", local_rank)

    print0(f"Linear probe best acc: {best_acc * 100:.2f}", local_rank)
    return {
        "best_acc": best_acc,
        "final_acc": epoch_accs[-1] if epoch_accs else 0.0,
    }
