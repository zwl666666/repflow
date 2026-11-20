#!/bin/bash
# FD post-training on JiT: the distilled student becomes a Frechet-Distance
# judge and only the generator (net) is updated.
#
# Usage:
#   bash recipe/jit_fd_posttrain.sh compute_stats   # once per student
#   bash recipe/jit_fd_posttrain.sh train
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUM_GPUS=${NUM_GPUS:-8}
IMAGENET_PATH=${IMAGENET_PATH:?set IMAGENET_PATH to ILSVRC/Data/CLS-LOC/train}
DISTILL_CKPT=${DISTILL_CKPT:?set DISTILL_CKPT to the distilled checkpoint.pth}
DISTILL_ARGS_JSON=${DISTILL_ARGS_JSON:-$(dirname "$DISTILL_CKPT")/args.json}
RESULTS_DIR=${RESULTS_DIR:-./outputs/jit_fd_posttrain}
REF_STATS=${REF_STATS:-${RESULTS_DIR}/student_ref_stats.npz}

FD_EPOCHS=${FD_EPOCHS:-100}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-1250}
TOTAL_STEPS=$((FD_EPOCHS * STEPS_PER_EPOCH))

case "${1:-train}" in
  compute_stats)
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
      "$ROOT/jit/compute_student_ref_stats.py" \
      --load_from "$DISTILL_CKPT" \
      --args_json "$DISTILL_ARGS_JSON" \
      --data_path "$IMAGENET_PATH" \
      --img_size 256 \
      --batch_size 128 \
      --num_workers "${NUM_WORKERS:-8}" \
      --output_path "$REF_STATS" \
      --fd_student_pool mean
    ;;
  train)
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
      "$ROOT/jit/main_jit_fd.py" \
      --load_from "$DISTILL_CKPT" \
      --args_json "$DISTILL_ARGS_JSON" \
      --fd_ref_stats "$REF_STATS" \
      --results_dir "$RESULTS_DIR" \
      --checkpoint_root "${CHECKPOINT_ROOT:-${RESULTS_DIR}/checkpoints}" \
      --global_batch_size "${GLOBAL_BATCH_SIZE:-1024}" \
      --lr 1e-5 \
      --lr_schedule cosine \
      --min_lr 0.0 \
      --total_steps "$TOTAL_STEPS" \
      --fd_warmup_steps 6250 \
      --queue_size 50000 \
      --fd_queue_fill_bsz 256 \
      --fd_eigvalsh \
      --fd_ema_beta 0.999 \
      --fd_num_sampling_steps 1 \
      --fd_eval_num_sampling_steps 1 \
      --fd_train_cfg 1.0 \
      --fd_eval_cfg 1.0 \
      --interval_min 0.0 \
      --interval_max 1.0 \
      --num_sampling_steps "${EVAL_SAMPLING_STEPS:-50}" \
      --save_every 2500 \
      --eval_every 2500 \
      --eval_during_train \
      --data_path "$IMAGENET_PATH" \
      --fid_ref_path "${FID_REF_PATH:?set FID_REF_PATH to VIRTUAL_imagenet256_labeled.npz}" \
      --evaluator_script "${EVALUATOR_SCRIPT:-$ROOT/sit/evaluator.py}" \
      --eval_temp_root "${EVAL_TEMP_ROOT:-/tmp/jit_fd_eval_tmp}" \
      --eval_num_fid_samples "${EVAL_NUM_FID_SAMPLES:-50000}" \
      --eval_per_proc_batch_size 128 \
      --gen_bsz 128 \
      --num_workers "${NUM_WORKERS:-8}" \
      --fd_student_pool mean \
      --print_freq 20
    ;;
  *)
    echo "Usage: bash recipe/jit_fd_posttrain.sh [compute_stats|train]" >&2
    exit 1
    ;;
esac
