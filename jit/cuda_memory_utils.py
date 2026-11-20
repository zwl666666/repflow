"""CUDA memory release helpers shared by the JiT evaluation paths."""

from __future__ import annotations

import gc
import logging

import torch


def release_eval_cuda_memory(log=None, reset_compile: bool = False):
    """Release cached GPU memory after an evaluation pass.

    Args:
        log: an optional ``logging.Logger`` or callable used to report the
            release. Rank > 0 workers pass ``None`` to stay silent.
        reset_compile: also reset ``torch.compile`` caches. Needed when the
            evaluation compiled representation models that the training loop
            does not reuse.
    """
    gc.collect()
    if reset_compile and hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if log is None:
        return
    message = "CUDA cache released after evaluation"
    if isinstance(log, logging.Logger):
        log.info(message)
    elif callable(log):
        try:
            log(message)
        except Exception:
            pass
