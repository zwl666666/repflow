"""Generate a fixed 30k COCO validation set from an MMDiT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel

from coco_samplers import euler_sampler
from mmdit_coco import MMDiT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP_L14 = PROJECT_ROOT / "hf_models" / "clip-vit-large-patch14"
DEFAULT_VAE = PROJECT_ROOT / "hf_models" / "sd-vae-ft-ema"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--state-key",
        type=str,
        default="generator_ema",
        choices=["generator_ema", "model"],
        help="Checkpoint state to sample from; use model to bypass EMA.",
    )
    parser.add_argument("--captions-file", type=str, required=True)
    parser.add_argument(
        "--manifest-file",
        type=str,
        default=None,
        help="Optional fixed manifest; otherwise one is generated from captions-file.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--clip-model",
        type=str,
        default=str(DEFAULT_CLIP_L14),
    )
    parser.add_argument("--vae-model", type=str, default=str(DEFAULT_VAE))
    parser.add_argument("--latent-scale", type=float, default=0.18215)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    return parser.parse_args()


def build_manifest(captions_file: str, num_images: int, seed: int):
    with open(captions_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    image_records = {record["id"]: record for record in data["images"]}
    captions_by_image = {}
    for annotation in data["annotations"]:
        captions_by_image.setdefault(int(annotation["image_id"]), []).append(
            str(annotation["caption"]).strip()
        )
    image_ids = sorted(image_id for image_id in image_records if captions_by_image.get(image_id))
    if num_images > len(image_ids):
        raise ValueError(f"Requested {num_images} images, but only {len(image_ids)} have captions")

    rng = np.random.default_rng(seed)
    selected_ids = rng.permutation(image_ids)[:num_images]
    manifest = []
    for image_id in selected_ids:
        captions = captions_by_image[int(image_id)]
        caption_index = int(rng.integers(0, len(captions)))
        image_seed = int(rng.integers(0, 2**31 - 1))
        manifest.append(
            {
                "image_id": int(image_id),
                "caption": captions[caption_index],
                "seed": image_seed,
            }
        )
    return manifest


def noise_from_seeds(seeds, shape, device):
    noises = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noises.append(torch.randn(shape, generator=generator))
    return torch.stack(noises).to(device)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    if args.manifest_file:
        with Path(args.manifest_file).open("r", encoding="utf-8") as handle:
            manifest = [json.loads(line) for line in handle if line.strip()]
        if accelerator.is_main_process:
            manifest_path.write_text(Path(args.manifest_file).read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(args.captions_file, args.num_images, args.seed)
        if accelerator.is_main_process:
            with manifest_path.open("w", encoding="utf-8") as handle:
                for record in manifest:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    accelerator.wait_for_everyone()

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    work_items = [
        (index, record)
        for index, record in enumerate(manifest)
        if index % world_size == rank
    ]

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = MMDiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        depth=24,
        mlp_ratio=4.0,
        context_dim=768,
        qkv_bias=True,
        learn_sigma=False,
    ).to(device)
    model.load_state_dict(checkpoint[args.state_key])
    model.eval()

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    text_encoder = CLIPTextModel.from_pretrained(args.clip_model).to(device)
    text_encoder.eval()
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()

    empty_tokens = tokenizer(
        [""],
        truncation=True,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    )
    empty_context = text_encoder(
        input_ids=empty_tokens.input_ids.to(device),
        attention_mask=empty_tokens.attention_mask.to(device),
    ).last_hidden_state

    total = len(work_items)
    for start in range(0, total, args.batch_size):
        batch = work_items[start : start + args.batch_size]
        indices = [index for index, _ in batch]
        records = [record for _, record in batch]
        captions = [record["caption"] for record in records]
        seeds = [record["seed"] for record in records]
        tokens = tokenizer(
            captions,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        context = text_encoder(
            input_ids=tokens.input_ids.to(device),
            attention_mask=tokens.attention_mask.to(device),
        ).last_hidden_state
        noise = noise_from_seeds(seeds, (4, 32, 32), device)
        current_null_context = empty_context.expand(context.shape[0], -1, -1)

        with accelerator.autocast():
            latents = euler_sampler(
                model,
                noise,
                context,
                current_null_context,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
            )
            decoded = vae.decode(latents / args.latent_scale).sample

        images = ((decoded.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        for offset, array in enumerate(images):
            image_index = indices[offset]
            Image.fromarray(array).save(images_dir / f"{image_index:08d}.png")

        if accelerator.is_main_process:
            print(f"Rank {rank} generated {min(start + args.batch_size, total)} / {total} images")


if __name__ == "__main__":
    main()
