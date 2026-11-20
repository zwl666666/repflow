# RepFlow

**RepFlow: Reciprocal Supervision Improves Generation and Representation in Flow Models**

This repository hosts the official PyTorch implementation of RepFlow, including both the latent-space (SiT / MMDiT) and pixel-space (JiT) realizations used in the paper, as well as the Fréchet-Distance (FD) post-training stage.

- Paper: [arXiv:2609.33217](https://arxiv.org/abs/2609.33217)

## Overview

Flow-matching generators are trained only through their generated velocity field, so their intermediate features receive no direct visual supervision. RepFlow adds a small student network next to the generator and supervises it reciprocally:

- **G2R (generator-to-representation):** the generator's intermediate features teach the student through a projector, using an MSE + cosine-alignment distillation loss.
- **R2G (representation-to-generator):** the student's representation is projected back and injected into the generator, so representation learning also improves generation.

The same recipe is realized on two very different backbones to show that the effect is architectural:

- **`sit/`** - latent-space flow models on ImageNet 256 (class-conditional and unconditional SiT-XL/2), plus a supplementary text-conditional COCO MMDiT.
- **`jit/`** - pixel-space flow models on ImageNet 256 (class-conditional and unconditional JiT-XL/16), including the FD post-training entry point.

## Repository Layout

```text
sit/     Latent-space implementations (SiT-XL/2 + COCO MMDiT)
jit/     Pixel-space implementations (JiT-XL/16 + FD post-training)
recipe/  Parameterized launch scripts for every experiment
```

Key entry points:

| Experiment | Trainer | Recipe |
|---|---|---|
| SiT, unconditional ImageNet | `sit/train.py` | `recipe/sit_uncond.sh` |
| SiT, class-conditional ImageNet | `sit/train.py` | `recipe/sit_cond.sh` |
| MMDiT, conditional COCO (supplementary) | `sit/train_mmdit_coco.py` | `recipe/sit_mmdit_coco.sh` |
| JiT, conditional / unconditional distill | `jit/train_distill.py` | `recipe/jit_distill.sh` |
| JiT, FD post-training | `jit/train_fd.py` | `recipe/jit_fd_posttrain.sh` |

## Installation

```bash
conda create -n repflow python=3.10 -y
conda activate repflow
pip install -r requirements.txt
```

For a full conda environment matching the training machines, use `jit/environment.yaml`.

## Data

**ImageNet 256** (`ILSVRC/Data/CLS-LOC/train`, ImageFolder layout) is required by the SiT and JiT recipes. Point the recipes at your copy:

```bash
export IMAGENET_PATH=/path/to/ILSVRC/Data/CLS-LOC/train
```

**COCO 2014** (train/val images plus caption annotations) is required by the conditional MMDiT recipe:

```bash
export COCO_ROOT=/path/to/coco2014   # expects train2014/, val2014/, annotations/
```

**Evaluation assets.** FID evaluation needs the ImageNet reference statistics and the TensorFlow evaluator:

- `VIRTUAL_imagenet256_labeled.npz` - place it in `assets/`
- `classify_image_graph_def.pb` - place it in `assets/`

Both are downloadable from the guided-diffusion release assets. Set `FID_REF_PATH` and `EVALUATOR_SCRIPT` if you keep them elsewhere.

## Training Recipes

Every recipe is parameterized through environment variables and defaults to an 8-GPU node. Scale `NUM_GPUS`, `GLOBAL_BATCH_SIZE`, and `PER_DEVICE_BATCH_SIZE` together for smaller machines.

### 1. SiT - Unconditional ImageNet 256

Trains SiT-XL/2 with a ViT-B student, patch masking (0.75), and a 10-block projector.

```bash
bash recipe/sit_uncond.sh
```

### 2. SiT - Class-Conditional ImageNet 256

The same SiT-XL/2 backbone with ImageNet class conditioning and classifier-free guidance (CFG=4) at evaluation.

```bash
bash recipe/sit_cond.sh
```

### 3. JiT - Class-Conditional / Unconditional Distillation

The default configuration is the paper's JiT-XL/16 teacher with a JiT-L/16 student and a 4-layer concat projector. Set `UNCONDITION=1` for the unconditional variant (which also switches the student to JiT-B/16).

```bash
bash recipe/jit_distill.sh

# unconditional variant
UNCONDITION=1 bash recipe/jit_distill.sh
```

### 4. JiT - FD Post-Training

After distillation, the student representation is used as a Fréchet-Distance judge and only the generator is updated. The recipe has two steps: compute the student reference statistics once, then run post-training.

```bash
bash recipe/jit_fd_posttrain.sh compute_stats
bash recipe/jit_fd_posttrain.sh train
```

### 5. Supplementary - MMDiT on Conditional COCO 256

Trains a compact MMDiT (depth 24, width 768) with a frozen CLIP text encoder and a frozen 4-channel VAE, using G2R + 0.1 x R2G.

```bash
bash recipe/sit_mmdit_coco.sh
```

## Evaluation

The training recipes run periodic FID automatically. To evaluate a checkpoint separately:

```bash
# SiT (latent space)
cd sit && python sample.py --model SiT-XL/2 --uncondition --ckpt /path/to/latest.pt

# MMDiT (COCO)
cd sit && python evaluate_coco_mmdit.py --ckpt /path/to/checkpoint.pt

# JiT (pixel space)
cd jit && python evaluate.py --ckpt /path/to/checkpoint.pth
```

Linear-probe utilities for representation quality are provided in `sit/linear_student.py` and `jit/linear_student.py`.

## Acknowledgements

This implementation builds on [SiT](https://github.com/willisma/SiT), [REPA](https://github.com/sihyun-yu/REPA), and [JiT](https://github.com/LTH14/JiT). We thank their authors for releasing their code.

## Citation

```bibtex
@article{zeng2026repflow,
  title={RepFlow: Reciprocal Supervision Improves Generation and Representation in Flow Models},
  author={Zeng, Weili and Tian, Feng and Liu, Shengqi and Yan, Yichao},
  journal={arXiv preprint arXiv:2609.33217},
  year={2026}
}
```
