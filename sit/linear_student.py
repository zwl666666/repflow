import argparse
import os
import PIL.Image as Image
from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models import AutoencoderKL
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from models import LatentStudentNet, SiT_models
from train_utils import center_crop_arr


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def print0(message: str, local_rank: int) -> None:
    if local_rank in (-1, 0):
        print(message, flush=True)


def as_torch_device(device) -> torch.device:
    """Accept torch.device, cuda index (train.py), or device string."""
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


def apply_checkpoint_args(args):
    ckpt_path = args.ckpt or args.resume
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        return args

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args")
    if ckpt_args is None:
        return args

    for name in [
        "model",
        "image_size",
        "num_classes",
        "latent_channels",
        "student_depth",
        "student_hidden_size",
        "student_mlp_ratio",
        "student_mask_ratio",
        "student_direct_image_input",
        "vae",
    ]:
        if hasattr(ckpt_args, name):
            setattr(args, name, getattr(ckpt_args, name))
    return args


def _strip_prefix_if_needed(state_dict, prefix: str):
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_student_state(student: nn.Module, ckpt_path: str, local_rank: int, use_student_ema: bool = True):
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if use_student_ema and "student_ema" in checkpoint and checkpoint["student_ema"] is not None:
        state_dict = checkpoint["student_ema"]
        weight_source = "student_ema"
    elif "student" in checkpoint:
        state_dict = checkpoint["student"]
        weight_source = "student"
    elif "model" in checkpoint:
        state_dict = {
            key[len("student."):]: value
            for key, value in checkpoint["model"].items()
            if key.startswith("student.")
        }
        if not state_dict:
            raise RuntimeError("Checkpoint does not contain a `student` state dict.")
        weight_source = "student (from model)"
    else:
        raise RuntimeError("Unsupported checkpoint format: missing `student`.")

    state_dict = _strip_prefix_if_needed(state_dict, "module.")
    missing, unexpected = student.load_state_dict(state_dict, strict=False)
    print0(f"Loaded {weight_source} weights from {ckpt_path}", local_rank)
    if missing:
        print0(f"Missing keys (first 20): {missing[:20]}", local_rank)
    if unexpected:
        print0(f"Unexpected keys (first 20): {unexpected[:20]}", local_rank)


def infer_patch_size(model_name: str, latent_size: int, latent_channels: int, num_classes: int) -> int:
    teacher = SiT_models[model_name](
        input_size=latent_size,
        in_channels=latent_channels,
        num_classes=num_classes,
        class_dropout_prob=0.0,
    )
    return teacher.patch_size


def get_student_input_config(args, teacher_patch_size: int) -> Tuple[int, int, int]:
    if args.student_direct_image_input:
        return args.image_size, teacher_patch_size * 8, 3
    return args.image_size // 8, teacher_patch_size, args.latent_channels


def build_student(args) -> LatentStudentNet:
    latent_size = args.image_size // 8
    patch_size = infer_patch_size(args.model, latent_size, args.latent_channels, args.num_classes)
    student_input_size, student_patch_size, student_in_channels = get_student_input_config(args, patch_size)
    return LatentStudentNet(
        input_size=student_input_size,
        patch_size=student_patch_size,
        in_channels=student_in_channels,
        hidden_size=args.student_hidden_size,
        depth=args.student_depth,
        mlp_ratio=args.student_mlp_ratio,
        mask_ratio=args.student_mask_ratio,
    )


class StudentFeatureExtractor:
    def __init__(self, student: nn.Module, vae: Optional[AutoencoderKL], device: torch.device, use_amp: bool, student_direct_image_input: bool):
        self.student = student
        self.vae = vae
        self.device = device
        self.student_direct_image_input = student_direct_image_input
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        images = images.float().div_(255.0)
        images = images.mul_(2.0).sub_(1.0)

        autocast_ctx = torch.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()
        with autocast_ctx:
            if self.student_direct_image_input:
                student_inputs = images
            else:
                student_inputs = self.vae.encode(images).latent_dist.sample().mul_(0.18215)
            tokens = self.student(student_inputs, use_mask=False)
        return _avg_pool_patches(tokens).float()


class LinearClassifier(nn.Module):
    def __init__(
        self,
        feature_extractor: StudentFeatureExtractor,
        feat_dim: int,
        num_classes: int,
        base_lr: float,
        total_epochs: int,
        local_rank: int,
        device: torch.device,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

        head = nn.Sequential(
            nn.BatchNorm1d(feat_dim, affine=False),
            nn.Linear(feat_dim, num_classes, bias=True),
        ).to(device)
        nn.init.normal_(head[1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(head[1].bias)

        if dist.is_available() and dist.is_initialized():
            # Use current CUDA device (set by train.py / torchrun), not global rank.
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
        self.optimizer.step()
        return loss.item()

    @torch.inference_mode()
    def test_step(self, images: torch.Tensor) -> torch.Tensor:
        self.classifier.eval()
        feats = self.feature_extractor(images)
        logits = self.classifier(feats)
        return logits.argmax(dim=-1)


@torch.inference_mode()
def infer_feat_dim(feature_extractor: StudentFeatureExtractor, image_size: int) -> int:
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.uint8, device="cpu")
    feats = feature_extractor(dummy)
    return feats.shape[1]


@torch.no_grad()
def test_acc(classifier: LinearClassifier, valid_loader, device: torch.device, local_rank: int) -> float:
    correct = 0
    total = 0
    for images, labels in tqdm(valid_loader, disable=(local_rank not in (-1, 0))):
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


def _prepare_student_for_probe(args, device: torch.device, local_rank: int, student=None):
    if student is None:
        student = build_student(args).to(device)
        load_student_state(
            student,
            args.ckpt or args.resume,
            local_rank,
            use_student_ema=getattr(args, "use_student_ema", True),
        )
    else:
        student = student.module if hasattr(student, "module") else student
    student = student.to(device).to(memory_format=torch.channels_last)
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)
    return student


def run_linear_probe(args, device, local_rank: int, student=None):
    """
    Train a linear classifier on frozen student features and return validation metrics.
    When `student` is provided, weights are taken in-process (e.g. from train.py DDP).
    """
    device = as_torch_device(device)
    use_amp = args.use_amp and device.type == "cuda"
    student = _prepare_student_for_probe(args, device, local_rank, student=student)

    vae = None
    if not args.student_direct_image_input:
        vae_source = args.vae_model or f"./sd-vae-ft-{args.vae}"
        vae = AutoencoderKL.from_pretrained(vae_source).to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad_(False)

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

    feature_extractor = StudentFeatureExtractor(
        student=student,
        vae=vae,
        device=device,
        use_amp=use_amp,
        student_direct_image_input=args.student_direct_image_input,
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
                pbar.set_description(f"[epoch {epoch} / iter {step}] lr={classifier.get_lr():.2e} loss={loss:.4f}")

        classifier.schedule_step()
        acc = test_acc(classifier, val_loader, device=device, local_rank=local_rank)
        epoch_accs.append(acc)
        best_acc = max(best_acc, acc)
        print0(f"[epoch {epoch}] acc: {acc * 100:.2f}", local_rank)

    print0(f"Best acc: {best_acc * 100:.2f}", local_rank)
    return {
        "best_acc": best_acc,
        "final_acc": epoch_accs[-1] if epoch_accs else 0.0,
        "epoch_accs": epoch_accs,
    }


def train(args, device: torch.device, local_rank: int):
    run_linear_probe(args, device=device, local_rank=local_rank)


def init_distributed(local_rank: int) -> Tuple[torch.device, int]:
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"local_rank {local_rank} >= available gpus {torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), local_rank

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        return device, -1

    return torch.device("cpu"), -1


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Local image root for the linear probe training set.",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        required=True,
        help="Local image root for the linear probe validation set.",
    )
    parser.add_argument("--train-label-file", type=str, default=None)
    parser.add_argument("--val-label-file", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None, help="Student checkpoint path saved by `train.py`.")
    parser.add_argument("--resume", type=str, default=None, help="Alias of --ckpt.")
    parser.add_argument(
        "--use-student-ema",
        action="store_true",
        default=True,
        help="Load student_ema from checkpoint when present (default).",
    )
    parser.add_argument(
        "--no-use-student-ema",
        action="store_false",
        dest="use_student_ema",
        help="Load live student weights instead of student_ema.",
    )
    parser.add_argument("--vae-model", type=str, default=None, help="Diffusers VAE repo id or local path.")

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    parser.add_argument("--student-depth", type=int, default=4)
    parser.add_argument("--student-hidden-size", type=int, default=384)
    parser.add_argument("--student-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--student-mask-ratio", type=float, default=0.75)
    parser.add_argument("--student-direct-image-input", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.add_argument("--use-amp", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.set_defaults(pin_mem=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args = apply_checkpoint_args(args)
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    device, local_rank = init_distributed(local_rank)
    try:
        train(args, device=device, local_rank=local_rank)
    finally:
        cleanup()
