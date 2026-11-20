<div align="center">

# RepFlow

**Reciprocal Supervision Improves Generation and Representation in Flow Models**

[![arXiv](https://img.shields.io/badge/arXiv-2609.33217-b31b1b.svg)](https://arxiv.org/abs/2609.33217)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](requirements.txt)

Weili Zeng &middot; Feng Tian &middot; Shengqi Liu &middot; Yichao Yan

</div>

## Overview

Flow-matching generators learn visual structure through their velocity field, but their internal states are entangled with noise level and network depth, so they do not directly provide a stable visual representation. **RepFlow** learns one from the generator's own evolving computation and uses it to guide generation:

```
masked clean image ─▶ timestep-free encoder ─▶ timestep-conditioned predictor ─▶ generator states
                        (student)                                          (across depth & noise level)
```

- **G2R** (generator &rarr; representation): generator states supervise the encoder and predictor, which must recover those states from a masked clean image without seeing the reference noise or the masked content.
- **R2G** (representation &rarr; generator): the learned representation is fed back to guide generation, and the improved generator in turn provides better supervision, forming a reciprocal learning process.
- **FD post-training:** the same representation serves as a distributional metric, a Fr&eacute;chet-Distance judge for one-step JiT post-training, extending its role from instance-level alignment to distribution-level supervision.

No externally pretrained representation teacher is used. Across the two unconditional settings (latent-space SiT and pixel-space JiT), FID decreases by **19.2&ndash;40.8%** relative to native training, while frozen linear-probe accuracy improves by **6.80&ndash;10.04** points over the best searched raw generator features.

## Contents

| Experiment | Trainer | Recipe |
|:---|:---|:---|
| SiT &middot; unconditional ImageNet 256 | [`sit/train.py`](sit/train.py) | [`recipe/sit_uncond.sh`](recipe/sit_uncond.sh) |
| SiT &middot; class-conditional ImageNet 256 | [`sit/train.py`](sit/train.py) | [`recipe/sit_cond.sh`](recipe/sit_cond.sh) |
| JiT &middot; conditional / unconditional distill | [`jit/train_distill.py`](jit/train_distill.py) | [`recipe/jit_distill.sh`](recipe/jit_distill.sh) |
| JiT &middot; FD post-training | [`jit/train_fd.py`](jit/train_fd.py) | [`recipe/jit_fd_posttrain.sh`](recipe/jit_fd_posttrain.sh) |
| MMDiT &middot; conditional COCO 256 (supplementary) | [`sit/train_mmdit_coco.py`](sit/train_mmdit_coco.py) | [`recipe/sit_mmdit_coco.sh`](recipe/sit_mmdit_coco.sh) |

```text
sit/      latent-space flow models (SiT-XL/2), plus a supplementary COCO MMDiT
jit/      pixel-space flow models (JiT-XL/16) with FD post-training
recipe/   parameterized launch scripts
```

## Installation

```bash
conda create -n repflow python=3.10 -y
conda activate repflow
pip install -r requirements.txt
```

For a full environment matching the training machines, use [`jit/environment.yaml`](jit/environment.yaml).

## Data

**ImageNet 256** (ImageFolder layout, `ILSVRC/Data/CLS-LOC/train`) is required by the SiT and JiT recipes:

```bash
export IMAGENET_PATH=/path/to/ILSVRC/Data/CLS-LOC/train
```

**COCO 2014** (images and caption annotations) is only needed by the supplementary MMDiT recipe:

```bash
export COCO_ROOT=/path/to/coco2014   # expects train2014/, val2014/, annotations/
```

<details>
<summary><b>Evaluation assets</b> (needed for periodic FID)</summary>

Place both files in `assets/`:

- `VIRTUAL_imagenet256_labeled.npz` &mdash; ImageNet reference statistics
- `classify_image_graph_def.pb` &mdash; Inception graph for the TensorFlow evaluator

Both are downloadable from the [guided-diffusion](https://github.com/openai/guided-diffusion) release assets. If you keep them elsewhere, point `FID_REF_PATH` and `EVALUATOR_SCRIPT` at them.

</details>

## Training

All recipes default to an 8-GPU node and are parameterized through environment variables. For smaller machines, scale `NUM_GPUS` and `GLOBAL_BATCH_SIZE` together.

### 1. SiT, unconditional ImageNet 256

SiT-XL/2 with a ViT-B student, patch masking ratio 0.75, and a 10-block projector.

```bash
bash recipe/sit_uncond.sh
```

### 2. SiT, class-conditional ImageNet 256

The same backbone and student, with ImageNet class conditioning (dropout 0.1) and classifier-free guidance (CFG=4) at evaluation.

```bash
bash recipe/sit_cond.sh
```

### 3. JiT, conditional / unconditional distillation

JiT-XL/16 generator with a JiT-L/16 student and a 4-layer concat projector. Set `UNCONDITION=1` for the unconditional variant, which also switches the student to JiT-B/16.

```bash
bash recipe/jit_distill.sh

# unconditional variant
UNCONDITION=1 bash recipe/jit_distill.sh
```

### 4. JiT, FD post-training

After distillation, the student representation becomes a Fr&eacute;chet-Distance judge and only the generator is updated. Compute the reference statistics once, then post-train:

```bash
bash recipe/jit_fd_posttrain.sh compute_stats
bash recipe/jit_fd_posttrain.sh train
```

### 5. Supplementary: MMDiT on conditional COCO 256

A compact MMDiT (depth 24, width 768) with a frozen CLIP text encoder and a frozen 4-channel VAE, trained with G2R + 0.1 &times; R2G.

```bash
bash recipe/sit_mmdit_coco.sh
```

## Evaluation

Training recipes run periodic FID automatically. To evaluate a checkpoint separately:

```bash
# SiT (latent space)
cd sit && python sample.py --model SiT-XL/2 --uncondition --ckpt /path/to/latest.pt

# MMDiT (COCO)
cd sit && python evaluate_coco_mmdit.py --ckpt /path/to/checkpoint.pt

# JiT (pixel space)
cd jit && python evaluate.py --ckpt /path/to/checkpoint.pth
```

Linear-probe utilities for representation quality are provided in [`sit/linear_student.py`](sit/linear_student.py) and [`jit/linear_student.py`](jit/linear_student.py).

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
