#!/bin/bash
# RepFlow on SiT-XL/2, class-conditional ImageNet 256, with a ViT-B student.
# Classifier-free guidance is enabled during evaluation (CFG=4).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUM_GPUS=${NUM_GPUS:-8}
IMAGENET_PATH=${IMAGENET_PATH:?set IMAGENET_PATH to ILSVRC/Data/CLS-LOC/train}
RESULTS_DIR=${RESULTS_DIR:-./outputs/sit_cond}
CKPT_DIR=${CKPT_DIR:-${RESULTS_DIR}/checkpoints}

torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  "$ROOT/sit/train.py" \
  --model SiT-XL/2 \
  --data-path "$IMAGENET_PATH" \
  --image-size 256 \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-1024}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --lr "${LR:-1e-4}" \
  --weight-decay 0 \
  --class-dropout-prob "${CLASS_DROPOUT_PROB:-0.1}" \
  --ckpt-every-epoch "${CKPT_EVERY_EPOCH:-50}" \
  --epochs "${EPOCHS:-50}" \
  --bf16 \
  --student-depth 12 \
  --student-hidden-size 768 \
  --student-feature-layers "5,12,19,26" \
  --student-loss-weight 1.0 \
  --student-align-cosine-weight 0.0 \
  --student-align-teacher-layer 7 \
  --student-mask-ratio 0.75 \
  --student-use-mask \
  --student-direct-image-input \
  --student-clean-ema \
  --projector-hidden-size 384 \
  --projector-depth 10 \
  --projector-mlp-ratio 4.0 \
  --repa-proj-coeff 0 \
  --results-dir "$RESULTS_DIR" \
  --ckpt-dir "$CKPT_DIR" \
  --eval-cfg-scale "${EVAL_CFG_SCALE:-4.0}"
