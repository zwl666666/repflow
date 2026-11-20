#!/bin/bash
# RepFlow distillation on JiT-XL/16 (pixel space), class-conditional ImageNet 256.
# Teacher JiT-XL/16, student JiT-L/16, 4-layer concat projector.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUM_GPUS=${NUM_GPUS:-8}
IMAGENET_PATH=${IMAGENET_PATH:?set IMAGENET_PATH to ILSVRC/Data/CLS-LOC/train}
RESULTS_DIR=${RESULTS_DIR:-./outputs/jit_cond}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${RESULTS_DIR}/checkpoints}

torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  "$ROOT/jit/train_distill.py" \
  --training_mode distill \
  --model JiT-XL/16 \
  --student_model JiT-L/16 \
  --projector_model JiT-L/16 \
  --teacher_feature_layers "5,12,19,26" \
  --teacher_cons_layer 7 \
  --student_mask_ratio 0.75 \
  --projector_block_dim "${PROJECTOR_BLOCK_DIM:-1024}" \
  --projector_num_blocks "${PROJECTOR_NUM_BLOCKS:-10}" \
  --projector_num_heads "${PROJECTOR_NUM_HEADS:-16}" \
  --projector_mlp_ratio 4.0 \
  --projector_out_dim "${PROJECTOR_OUT_DIM:-4608}" \
  --projector_num_layers "${PROJECTOR_NUM_LAYERS:-4}" \
  --flow_loss_weight "${FLOW_LOSS_WEIGHT:-1.0}" \
  --distill_loss_weight "${DISTILL_LOSS_WEIGHT:-1.0}" \
  --distill_mse_weight "${DISTILL_MSE_WEIGHT:-1.0}" \
  --distill_cosine_weight "${DISTILL_COSINE_WEIGHT:-0.1}" \
  --P_mean -0.8 --P_std 0.8 \
  --img_size 256 --noise_scale 1.0 \
  --global_batch_size "${GLOBAL_BATCH_SIZE:-1024}" \
  --blr "${BLR:-5e-5}" \
  --lr_schedule cosine --min_lr 1e-6 \
  --epochs "${EPOCHS:-2000}" --warmup_epochs 5 \
  --ema_update_freq 1 \
  --ckpt_every_epoch "${CKPT_EVERY_EPOCH:-40}" \
  --results_dir "$RESULTS_DIR" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --eval_during_train \
  --eval_num_fid_samples "${EVAL_NUM_FID_SAMPLES:-50000}" \
  --eval_per_proc_batch_size 128 \
  --eval_temp_root "${EVAL_TEMP_ROOT:-/tmp/jit_eval_tmp}" \
  --fid_ref_path "${FID_REF_PATH:?set FID_REF_PATH to VIRTUAL_imagenet256_labeled.npz}" \
  --evaluator_script "${EVALUATOR_SCRIPT:-$ROOT/sit/evaluator.py}" \
  --cfg 1.0 \
  --gen_bsz 128 \
  --data_path "$IMAGENET_PATH" \
  --num_workers "${NUM_WORKERS:-8}"
