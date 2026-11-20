#!/bin/bash
# Supplementary: RepFlow on a compact MMDiT, text-conditional COCO 256
# (frozen CLIP text encoder + frozen 4-channel VAE).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUM_GPUS=${NUM_GPUS:-8}
COCO_ROOT=${COCO_ROOT:?set COCO_ROOT to your coco2014 directory}
CLIP_MODEL_PATH=${CLIP_MODEL_PATH:?set CLIP_MODEL_PATH to a CLIP-ViT-L/14 checkpoint}

IMAGE_DIR="$COCO_ROOT/train2014"
CAPTIONS_FILE="$COCO_ROOT/annotations/captions_train2014.json"
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/sit_mmdit_coco}

accelerate launch --num_processes "$NUM_GPUS" --multi_gpu --mixed_precision bf16 \
  "$ROOT/sit/train_mmdit_coco.py" \
  --method repflow \
  --image-dir "$IMAGE_DIR" \
  --captions-file "$CAPTIONS_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --resolution 256 \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-256}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --max-train-steps "${MAX_TRAIN_STEPS:-150000}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-50000}" \
  --save-every "${SAVE_EVERY:-50000}" \
  --log-every 100 \
  --lr "${LR:-1e-4}" \
  --clip-model "$CLIP_MODEL_PATH" \
  --caption-dropout 0.1
