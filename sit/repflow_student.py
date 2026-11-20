"""RepFlow student, predictor, and R2G projector used with COCO MMDiT."""

from __future__ import annotations

import torch
import torch.nn as nn

from models import LatentProjector, LatentStudentNet


class RepFlowStudent(LatentStudentNet):
    """LatentStudentNet with an isolated RNG for patch masking."""

    def __init__(self, *args, mask_generator_device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_generator = torch.Generator(device=mask_generator_device)
        self.mask_generator.manual_seed(2027)

    def random_masking(self, x):
        batch_size, num_tokens, dim = x.shape
        num_keep = max(1, int(round(num_tokens * (1.0 - self.mask_ratio))))
        scores = torch.rand(
            batch_size,
            num_tokens,
            device=x.device,
            generator=self.mask_generator,
        )
        keep_idx = torch.topk(scores, k=num_keep, dim=1, largest=False).indices
        mask = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        mask.scatter_(1, keep_idx, False)
        x = x.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, dim))
        return x, mask, keep_idx


class R2GProjector(nn.Module):
    """Two-layer MLP used for generator-to-EMA-student cosine alignment."""

    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_repflow_modules(
    image_size=256,
    student_depth=24,
    student_hidden_size=1024,
    student_heads=16,
    generator_hidden_size=768,
    mask_ratio=0.75,
    predictor_depth=10,
    predictor_hidden_size=384,
    predictor_heads=12,
    teacher_hidden_size=3072,
    device="cpu",
):
    student = RepFlowStudent(
        input_size=image_size,
        patch_size=16,
        in_channels=3,
        hidden_size=student_hidden_size,
        depth=student_depth,
        num_heads=student_heads,
        mask_ratio=mask_ratio,
        mask_generator_device=device,
    ).to(device)
    predictor = LatentProjector(
        input_size=image_size,
        patch_size=16,
        student_hidden_size=student_hidden_size,
        teacher_hidden_size=teacher_hidden_size,
        projector_hidden_size=predictor_hidden_size,
        depth=predictor_depth,
        num_heads=predictor_heads,
    ).to(device)
    r2g_projector = R2GProjector(
        input_dim=generator_hidden_size,
        hidden_dim=1024,
        output_dim=student_hidden_size,
    ).to(device)
    return student, predictor, r2g_projector
