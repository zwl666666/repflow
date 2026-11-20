#!/usr/bin/env python3
"""
Bootstrap script for guided-diffusion evaluator.py.

Usage:
    python evaluator.py --download-only
    python evaluator.py <ref_batch.npz> <sample_batch.npz>
"""

import os
import sys
import urllib.request
from pathlib import Path


EVALUATOR_URL = (
    "https://raw.githubusercontent.com/openai/guided-diffusion/main/evaluations/evaluator.py"
)


def ensure_impl(script_dir: Path) -> Path:
    """
    Ensure the real OpenAI evaluator implementation exists locally.
    """
    impl_path = script_dir / "_guided_diffusion_evaluator_impl.py"
    if impl_path.is_file():
        return impl_path

    tmp_path = script_dir / (impl_path.name + ".tmp")
    print(f"[evaluator-bootstrap] Downloading evaluator from: {EVALUATOR_URL}")
    with urllib.request.urlopen(EVALUATOR_URL, timeout=60) as resp:
        content = resp.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    os.replace(tmp_path, impl_path)
    print(f"[evaluator-bootstrap] Saved evaluator implementation to: {impl_path}")
    return impl_path


def main():
    script_dir = Path(__file__).resolve().parent
    impl_path = ensure_impl(script_dir)

    if len(sys.argv) >= 2 and sys.argv[1] == "--download-only":
        print(f"[evaluator-bootstrap] Download-only complete: {impl_path}")
        return

    # Forward all args to the real evaluator.py process.
    cmd = [sys.executable, str(impl_path), *sys.argv[1:]]
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
