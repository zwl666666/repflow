"""Build ImageNet train dataset from S3 (ImageFolder layout) via Petrel SDK."""

from __future__ import annotations

import os
import pickle
from typing import Any

import torch.distributed as dist

from s3_imagefolder import S3ImageFolder
from util import misc


def add_s3_dataset_args(parser):
    parser.add_argument(
        "--s3_endpoint",
        default=None,
        type=str,
        help="Deprecated (ignored). S3 reads use Petrel SDK via PETRELOSS_CONF / cfs/petreloss.conf.",
    )
    parser.add_argument(
        "--s3_region",
        default="us-east-1",
        type=str,
        help="Deprecated (ignored). Kept for backward-compatible CLI.",
    )
    parser.add_argument(
        "--s3_index_cache",
        default=None,
        type=str,
        help="Optional shared path for pickle index; rank 0 builds once under distributed training.",
    )
    parser.add_argument(
        "--petrel_conf",
        default=None,
        type=str,
        help="Path to petreloss.conf; default PETRELOSS_CONF or $ZWL_ROOT/cfs/petreloss.conf.",
    )


def build_imagenet_train_folder(args: Any, transform, local_subdir: str = ""):
    """Local CLS-LOC/train dir or s3://.../CLS-LOC/train prefix."""
    data_path = str(args.data_path)
    if not data_path.startswith("s3://"):
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Local --data_path not found: {data_path}")
        from torchvision.datasets import ImageFolder

        return ImageFolder(data_path, transform=transform)

    cache = getattr(args, "s3_index_cache", None) or None
    petrel_conf = getattr(args, "petrel_conf", None) or None
    idx = None
    if cache:
        if misc.is_dist_avail_and_initialized():
            if misc.get_rank() == 0 and not os.path.isfile(cache):
                blob = S3ImageFolder.build_index(data_path, petrel_conf=petrel_conf)
                d = os.path.dirname(cache)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(cache, "wb") as f:
                    pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
            dist.barrier()
            with open(cache, "rb") as f:
                idx = pickle.load(f)
        else:
            if not os.path.isfile(cache):
                blob = S3ImageFolder.build_index(data_path, petrel_conf=petrel_conf)
                d = os.path.dirname(cache)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(cache, "wb") as f:
                    pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cache, "rb") as f:
                idx = pickle.load(f)
        return S3ImageFolder(
            data_path,
            transform=transform,
            index=idx,
            petrel_conf=petrel_conf,
        )
    return S3ImageFolder(data_path, transform=transform, petrel_conf=petrel_conf)
