"""Precompute CLIP text contexts for every COCO training caption."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

from coco_text_dataset import CocoTextDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "coco2014" if (PROJECT_ROOT / "coco2014").exists() else PROJECT_ROOT / "coco"
DEFAULT_CLIP_L14 = PROJECT_ROOT / "hf_models" / "clip-vit-large-patch14"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captions-file",
        type=str,
        default=str(DEFAULT_DATA_ROOT / "annotations" / "captions_train2014.json"),
    )
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_DATA_ROOT / "train2014"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "eval_assets" / "coco_train2014_all_contexts"),
    )
    parser.add_argument("--clip-model", type=str, default=str(DEFAULT_CLIP_L14))
    parser.add_argument("--caption-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def encode_captions(
    model,
    tokenizer,
    captions,
    device,
):
    tokens = tokenizer(
        captions,
        truncation=True,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return output


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CocoTextDataset(
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        caption_dropout=args.caption_dropout,
        seed=args.seed,
    )
    all_captions = []
    offsets = [0]
    for _, captions in dataset.records:
        all_captions.extend(captions)
        offsets.append(len(all_captions))
    total = len(all_captions)
    context_path = output_dir / "contexts.npy"
    contexts = np.lib.format.open_memmap(
        context_path,
        mode="w+",
        dtype=np.float16,
        shape=(total, 77, 768),
    )

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    model = CLIPTextModel.from_pretrained(args.clip_model).to(args.device)
    model.eval()

    for start in tqdm(range(0, total, args.batch_size), desc="Train caption contexts"):
        end = min(start + args.batch_size, total)
        captions = all_captions[start:end]
        output = encode_captions(model, tokenizer, captions, args.device)
        contexts[start:end] = output.detach().cpu().to(torch.float16).numpy()

    contexts.flush()
    np.save(output_dir / "offsets.npy", np.asarray(offsets, dtype=np.int64))

    empty_context = encode_captions(model, tokenizer, [""], args.device)
    empty_context_path = output_dir / "empty_context.npy"
    np.save(empty_context_path, empty_context.detach().cpu().to(torch.float16).numpy())

    metadata = {
        "num_images": len(dataset),
        "num_captions": total,
        "seed": args.seed,
        "caption_dropout": args.caption_dropout,
        "clip_model": str(Path(args.clip_model).resolve()),
        "captions_sha256": sha256_file(Path(args.captions_file)),
        "context_shape": [total, 77, 768],
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
