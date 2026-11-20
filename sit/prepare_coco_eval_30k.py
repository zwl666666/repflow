"""Precompute the fixed COCO-30K evaluation manifest, FID stats, and CLIP text features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import tensorflow.compat.v1 as tf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

import _guided_diffusion_evaluator_impl as guided_evaluator
from sample_coco_mmdit import build_manifest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "coco2014" if (PROJECT_ROOT / "coco2014").exists() else PROJECT_ROOT / "coco"
DEFAULT_CLIP_B32 = PROJECT_ROOT / "hf_models" / "clip-vit-base-patch32"
DEFAULT_FID_GRAPH = Path(__file__).resolve().parent / "classify_image_graph_def.pb"


class ReferenceImageDataset(Dataset):
    def __init__(self, image_dir: Path, captions_file: Path, manifest_file: Path):
        with captions_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.image_records = {record["id"]: record for record in data["images"]}
        with manifest_file.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        self.image_dir = image_dir

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-file", type=str, default=str(DEFAULT_DATA_ROOT / "annotations" / "captions_val2014.json"))
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_DATA_ROOT / "val2014"))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "eval_assets" / "coco_val2014_30k"))
    parser.add_argument("--num-images", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--clip-model", type=str, default=str(DEFAULT_CLIP_B32))
    parser.add_argument("--fid-graph", type=str, default=str(DEFAULT_FID_GRAPH))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compute_reference_fid_stats(args, dataset):
    guided_evaluator.INCEPTION_V3_PATH = str(Path(args.fid_graph).resolve())
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        evaluator = guided_evaluator.Evaluator(session)
        batches = []
        for batch in tqdm(loader, desc="Reference images"):
            array = batch.numpy().transpose(0, 2, 3, 1) * 255.0
            batches.append(array.astype(np.float32))
        activations = evaluator.compute_activations(batches)
        stats = evaluator.compute_statistics(activations[0])
        spatial_stats = evaluator.compute_statistics(activations[1])
        return {
            "mu": stats.mu,
            "sigma": stats.sigma,
            "mu_s": spatial_stats.mu,
            "sigma_s": spatial_stats.sigma,
        }


def compute_clip_text_features(args, manifest):
    model = CLIPModel.from_pretrained(args.clip_model).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    features = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(manifest), args.batch_size), desc="CLIP text"):
            captions = [record["caption"] for record in manifest[start : start + args.batch_size]]
            inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            if not torch.is_tensor(text_features):
                text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            features.append(text_features.cpu().to(torch.float16).numpy())
    return np.concatenate(features, axis=0)


def main():
    args = parse_args()
    os.chdir(Path(__file__).resolve().parent)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    manifest = build_manifest(args.captions_file, args.num_images, args.seed)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    dataset = ReferenceImageDataset(Path(args.image_dir), Path(args.captions_file), manifest_path)
    fid_stats = compute_reference_fid_stats(args, dataset)
    fid_stats_path = output_dir / "fid_stats.npz"
    np.savez_compressed(fid_stats_path, **fid_stats)

    clip_features = compute_clip_text_features(args, manifest)
    clip_features_path = output_dir / "clip_text_features.npy"
    np.save(clip_features_path, clip_features)

    metadata = {
        "num_images": len(manifest),
        "seed": args.seed,
        "manifest_sha256": sha256_file(manifest_path),
        "fid_graph": str(Path(args.fid_graph).resolve()),
        "clip_model": str(Path(args.clip_model).resolve()),
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
