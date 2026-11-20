"""Compute FID-30K and CLIP cosine x100 for generated COCO images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError as exc:
    raise ImportError(
        "evaluate_coco_mmdit.py requires torchmetrics. Install it with `pip install torchmetrics`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_B32 = PROJECT_ROOT / "hf_models" / "clip-vit-base-patch32"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--reference-image-dir", type=str, required=True)
    parser.add_argument("--captions-file", type=str, required=True)
    parser.add_argument("--manifest-file", type=str, default=None)
    parser.add_argument("--eval-assets-dir", type=str, default=None)
    parser.add_argument(
        "--fid-graph",
        type=str,
        default=str(Path(__file__).resolve().parent / "classify_image_graph_def.pb"),
    )
    parser.add_argument("--output-file", type=str, default="metrics.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--clip-model",
        type=str,
        default=str(DEFAULT_CLIP_B32),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


class GeneratedImageDataset(Dataset):
    def __init__(self, image_dir: Path):
        self.paths = sorted(image_dir.glob("*.png"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB").resize((256, 256), Image.LANCZOS)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)


class ReferenceImageDataset(Dataset):
    def __init__(self, image_dir: Path, captions_file: Path, manifest_file: Path):
        with captions_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        image_records = {record["id"]: record for record in data["images"]}
        with manifest_file.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.image_dir = image_dir
        self.image_records = image_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_id = int(record["image_id"])
        file_name = str(self.image_records[image_id]["file_name"])
        image = Image.open(self.image_dir / file_name).convert("RGB")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side)).resize((256, 256), Image.LANCZOS)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)


def compute_clip_score(images_dir, manifest_file, model_name, device, batch_size, text_features=None):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    with manifest_file.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    similarities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            paths = [images_dir / f"{start + offset:08d}.png" for offset in range(len(batch))]
            images = [Image.open(path).convert("RGB") for path in paths]
            captions = [record["caption"] for record in batch]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            if not torch.is_tensor(image_features):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if text_features is None:
                text_inputs = processor(
                    text=captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
                batch_text_features = model.get_text_features(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                )
                if not torch.is_tensor(batch_text_features):
                    batch_text_features = batch_text_features.pooler_output
                batch_text_features = batch_text_features / batch_text_features.norm(
                    dim=-1, keepdim=True
                )
            else:
                batch_text_features = text_features[start : start + len(batch)].to(device)
            similarity = (image_features * batch_text_features).sum(dim=-1)
            similarities.extend(similarity.cpu().tolist())
    return 100.0 * float(np.mean(similarities))


def compute_metrics(
    images_dir,
    reference_image_dir,
    captions_file,
    manifest_file,
    device,
    batch_size=64,
    clip_model="openai/clip-vit-base-patch32",
    num_workers=8,
    fid_stats=None,
    clip_text_features=None,
    fid_graph=None,
):
    images_dir = Path(images_dir)
    reference_image_dir = Path(reference_image_dir)
    captions_file = Path(captions_file)
    manifest_file = Path(manifest_file)

    generated_dataset = GeneratedImageDataset(images_dir)
    reference_dataset = ReferenceImageDataset(reference_image_dir, captions_file, manifest_file)
    if len(generated_dataset) != len(reference_dataset):
        raise ValueError(
            f"Generated/reference count mismatch: {len(generated_dataset)} vs {len(reference_dataset)}"
        )

    generated_loader = DataLoader(generated_dataset, batch_size=batch_size, num_workers=num_workers)
    if fid_stats is not None:
        import tensorflow.compat.v1 as tf

        import _guided_diffusion_evaluator_impl as guided_evaluator

        guided_evaluator.INCEPTION_V3_PATH = str(Path(fid_graph).resolve())
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            evaluator = guided_evaluator.Evaluator(session)
            batches = []
            for batch in generated_loader:
                array = batch.numpy().transpose(0, 2, 3, 1) * 255.0
                batches.append(array.astype(np.float32))
            activations = evaluator.compute_activations(batches)
            sample_stats = evaluator.compute_statistics(activations[0])
            with np.load(fid_stats) as stats:
                reference_stats = guided_evaluator.FIDStatistics(
                    stats["mu"],
                    stats["sigma"],
                )
            fid_value = float(sample_stats.frechet_distance(reference_stats))
    else:
        reference_loader = DataLoader(
            reference_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        with torch.no_grad():
            for batch in reference_loader:
                fid.update(batch.to(device), real=True)
            for batch in generated_loader:
                fid.update(batch.to(device), real=False)
        fid_value = float(fid.compute().item())

    clip_score = compute_clip_score(
        images_dir,
        manifest_file,
        clip_model,
        device,
        batch_size,
        text_features=clip_text_features,
    )
    return {
        "num_images": len(generated_dataset),
        "fid": fid_value,
        "clip_cosine_x100": clip_score,
    }


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    manifest_file = Path(args.manifest_file or images_dir.parent / "manifest.jsonl")
    reference_dir = Path(args.reference_image_dir)
    captions_file = Path(args.captions_file)
    fid_stats = None
    clip_text_features = None
    if args.eval_assets_dir:
        assets_dir = Path(args.eval_assets_dir)
        candidate_fid_stats = assets_dir / "fid_stats.npz"
        candidate_clip_features = assets_dir / "clip_text_features.npy"
        if candidate_fid_stats.exists():
            fid_stats = str(candidate_fid_stats)
        if candidate_clip_features.exists():
            clip_text_features = torch.from_numpy(np.load(candidate_clip_features))

    metrics = compute_metrics(
        images_dir,
        reference_dir,
        captions_file,
        manifest_file,
        args.device,
        args.batch_size,
        clip_model=args.clip_model,
        num_workers=args.num_workers,
        fid_stats=fid_stats,
        clip_text_features=clip_text_features,
        fid_graph=args.fid_graph,
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
