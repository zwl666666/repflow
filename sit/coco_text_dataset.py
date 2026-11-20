"""COCO text-image dataset with deterministic center-square preprocessing."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoTextDataset(Dataset):
    """Return ``(image, caption)`` pairs for COCO 2014-style annotation JSON files."""

    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        image_size: int = 256,
        caption_dropout: float = 0.1,
        empty_caption: str = "",
        seed: int = 0,
        context_file: str | None = None,
        verify_images: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.captions_file = Path(captions_file)
        self.image_size = image_size
        self.caption_dropout = float(caption_dropout)
        self.empty_caption = empty_caption
        self.seed = seed
        self.contexts = None
        self.context_offsets = None
        self.empty_context = None

        with self.captions_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        image_records = {record["id"]: record for record in data["images"]}
        captions_by_image: Dict[int, List[str]] = {}
        for annotation in data["annotations"]:
            image_id = int(annotation["image_id"])
            caption = str(annotation["caption"]).strip()
            if caption:
                captions_by_image.setdefault(image_id, []).append(caption)

        self.records: List[Tuple[str, List[str]]] = []
        for image_id, record in image_records.items():
            captions = captions_by_image.get(image_id, [])
            if not captions:
                continue
            file_name = str(record["file_name"])
            image_path = self.image_dir / file_name
            if not verify_images or image_path.exists():
                self.records.append((str(image_path), captions))

        if context_file is not None:
            context_path = Path(context_file)
            offsets_path = context_path.parent / "offsets.npy"
            empty_context_path = context_path.parent / "empty_context.npy"
            if not offsets_path.exists() or not empty_context_path.exists():
                raise ValueError(
                    "A one-context-per-image cache fixes caption selection and dropout, "
                    "which causes image-caption pair memorization. Precompute all caption "
                    "contexts with prepare_coco_train_context.py, or omit context_file and "
                    "encode captions on the fly."
                )
            self.contexts = np.load(context_path, mmap_mode="r")
            self.context_offsets = np.load(offsets_path, mmap_mode="r")
            self.empty_context = np.load(empty_context_path, mmap_mode="r")
            if len(self.context_offsets) != len(self.records) + 1:
                raise ValueError(
                    f"Context offsets length {len(self.context_offsets)} does not match "
                    f"dataset length {len(self.records)} + 1"
                )
            if int(self.context_offsets[0]) != 0 or int(self.context_offsets[-1]) != len(self.contexts):
                raise ValueError("Context offsets do not span the complete context array")
            if self.contexts.ndim != 3 or self.contexts.shape[1:] != (77, 768):
                raise ValueError(
                    f"Context file must have shape [N, 77, 768], got {self.contexts.shape}"
                )
            if self.empty_context.shape != (77, 768):
                raise ValueError(
                    f"Empty context must have shape [77, 768], got {self.empty_context.shape}"
                )

        if not self.records:
            raise RuntimeError(
                f"No COCO image/caption pairs found under {self.image_dir} "
                f"with annotations {self.captions_file}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, captions = self.records[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self._center_square_resize(image, self.image_size)
            image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
            image = torch.from_numpy(image).div_(127.5).sub_(1.0)

        if self.contexts is not None:
            if self.caption_dropout > 0 and random.random() < self.caption_dropout:
                context = self.empty_context
            else:
                start = int(self.context_offsets[index])
                end = int(self.context_offsets[index + 1])
                if end <= start:
                    raise RuntimeError(f"Image index {index} has no caption contexts")
                context_index = random.randint(start, end - 1)
                context = self.contexts[context_index]
            context = torch.from_numpy(np.array(context, dtype=np.float32))
            return image, context

        caption = self.caption_for_index(index)

        return image, caption

    def caption_for_index(self, index: int) -> str:
        _, captions = self.records[index]
        if self.caption_dropout > 0:
            if random.random() < self.caption_dropout:
                return self.empty_caption
        return random.choice(captions)

    @staticmethod
    def _center_square_resize(image: Image.Image, image_size: int) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((image_size, image_size), Image.LANCZOS)
