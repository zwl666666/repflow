# S3-backed dataset mirroring torchvision.datasets.ImageFolder layout (one class = one subfolder).
# Credentials and list() use Petrel SDK (petrel_client) per Petrel_SDK.pdf / petreloss.conf.
# Hot-path get() in auto mode: boto3 first (fast) with botocore adaptive retries, then Petrel fallback (stable).

from __future__ import annotations

import io
import logging
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_CONF_CACHE: Dict[str, Dict[str, str]] = {}
_BOTO3_CLIENTS: Dict[str, Any] = {}

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
    ".JPEG",
    ".JPG",
)


def _s3_read_max_retries() -> int:
    return max(1, int(os.environ.get("S3_READ_RETRIES", "3")))


def _s3_boto_max_attempts() -> int:
    return max(3, int(os.environ.get("S3_BOTO_MAX_ATTEMPTS", "10")))


def _s3_petrel_fallback_enabled() -> bool:
    return os.environ.get("S3_BOTO_FALLBACK_PETREL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _s3_read_backend() -> str:
    backend = os.environ.get("S3_READ_BACKEND", "auto").strip().lower()
    if backend in ("auto", "boto3", "petrel"):
        return backend
    return "auto"


def _petrel_conf_path() -> str:
    env_path = os.environ.get("PETRELOSS_CONF", "").strip()
    if env_path:
        return env_path
    zwl = os.environ.get("ZWL_ROOT", "")
    project_conf = os.path.join(zwl, "cfs", "petreloss.conf")
    if os.path.isfile(project_conf):
        return project_conf
    return os.path.expanduser("~/petreloss.conf")


def parse_s3_url(s3_path: str) -> Tuple[str, str]:
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Expected s3:// URL, got: {s3_path!r}")
    rest = s3_path[5:]
    slash = rest.find("/")
    if slash == -1:
        return rest, ""
    bucket = rest[:slash]
    prefix = rest[slash + 1 :]
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    return bucket, prefix


def _s3_uri(bucket: str, key: str) -> str:
    key = key.lstrip("/")
    if key.startswith("s3://"):
        return key
    if key.startswith(f"{bucket}/"):
        return f"s3://{key}"
    return f"s3://{bucket}/{key}"


def _resolve_conf_path(conf_path: Optional[str] = None) -> str:
    path = os.path.abspath(os.path.expanduser(conf_path or _petrel_conf_path()))
    if not os.path.isfile(path):
        raise RuntimeError(
            f"Petrel config not found: {path}. Set PETRELOSS_CONF or place petreloss.conf "
            f"under $ZWL_ROOT/cfs/ or ~/."
        )
    return path


def _parse_petreloss_conf(conf_path: str) -> Dict[str, str]:
    path = _resolve_conf_path(conf_path)
    if path in _CONF_CACHE:
        return _CONF_CACHE[path]

    sections: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                sections.setdefault(current, {})
                continue
            if current is None or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            sections[current][key.lower()] = value

    default_section = sections.get("DEFAULT", sections.get("default", {}))
    cluster_name = default_section.get("default_cluster", "cluster1")
    cluster_cfg = sections.get(cluster_name, sections.get("cluster1", {}))
    if not cluster_cfg:
        raise RuntimeError(f"No cluster section found in petreloss.conf: {path}")

    _CONF_CACHE[path] = cluster_cfg
    return cluster_cfg


def _petrel_client(conf_path: Optional[str] = None) -> Any:
    from petrel_client.client import Client

    return Client(_resolve_conf_path(conf_path))


def _get_boto3_s3_client(conf_path: Optional[str] = None) -> Any:
    import boto3
    from botocore.config import Config

    path = _resolve_conf_path(conf_path)
    client = _BOTO3_CLIENTS.get(path)
    if client is not None:
        return client

    cfg = _parse_petreloss_conf(path)
    timeout_ms = int(cfg.get("request_timeout", "3000"))
    address_virtual = cfg.get("address_style", "").lower() == "virtual"
    use_sigv2 = cfg.get("use_sigv2", "").lower() in ("true", "1", "yes")
    max_pool = max(4, int(os.environ.get("S3_BOTO_MAX_POOL", "16")))
    boto_attempts = _s3_boto_max_attempts()

    client = boto3.client(
        "s3",
        endpoint_url=cfg["host_base"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        config=Config(
            signature_version="s3" if use_sigv2 else "s3v4",
            s3={"addressing_style": "virtual" if address_virtual else "path"},
            connect_timeout=10,
            read_timeout=max(timeout_ms / 1000.0, 30.0),
            max_pool_connections=max_pool,
            retries={"max_attempts": boto_attempts, "mode": "adaptive"},
        ),
    )
    _BOTO3_CLIENTS[path] = client
    return client


def _reset_boto3_client(conf_path: Optional[str] = None) -> None:
    path = _resolve_conf_path(conf_path)
    _BOTO3_CLIENTS.pop(path, None)


def _is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, ConnectionResetError, BrokenPipeError)):
        return True
    msg = str(exc).lower()
    return any(token in msg for token in ("connection", "reset", "broken", "closed"))


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, ConnectionResetError, BrokenPipeError, OSError)):
        return True
    cause = exc.__cause__
    if cause is not None and cause is not exc and _is_retryable_error(cause):
        return True
    msg = str(exc).lower()
    return any(token in msg for token in ("timeout", "connection", "incomplete", "reset", "closed"))


def _read_object_bytes_petrel(dataset: Any, bucket: str, key: str) -> bytes:
    uri = _s3_uri(bucket, key)
    max_retries = _s3_read_max_retries()
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            data = dataset._get_client().get(uri)
            if data is None:
                raise RuntimeError(f"Empty response for {uri}")
            return data
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise
            last_error = exc
            dataset._client = None
            if attempt + 1 >= max_retries:
                break
            delay = min(2**attempt, 30)
            logger.warning(
                "Petrel S3 read failed for %s (attempt %d/%d): %s; retry in %ds",
                uri,
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _read_object_bytes_boto3(dataset: Any, bucket: str, key: str) -> bytes:
    """boto3 with botocore adaptive retries; outer loop only for hard failures."""
    max_retries = _s3_read_max_retries()
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            client = _get_boto3_s3_client(dataset.petrel_conf)
            response = client.get_object(Bucket=bucket, Key=key)
            data = response["Body"].read()
            if not data:
                raise RuntimeError(f"Empty response for s3://{bucket}/{key}")
            return data
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise
            last_error = exc
            if _is_connection_error(exc):
                _reset_boto3_client(dataset.petrel_conf)
            if attempt + 1 >= max_retries:
                break
            delay = min(2**attempt, 10) + random.uniform(0, 0.5)
            logger.warning(
                "boto3 S3 read failed for s3://%s/%s (attempt %d/%d): %s; retry in %.1fs",
                bucket,
                key,
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _read_object_bytes(dataset: Any, bucket: str, key: str) -> bytes:
    backend = _s3_read_backend()
    if backend == "petrel":
        return _read_object_bytes_petrel(dataset, bucket, key)
    if backend == "boto3":
        return _read_object_bytes_boto3(dataset, bucket, key)

    # auto: boto3 first (fast), Petrel fallback (stable)
    try:
        return _read_object_bytes_boto3(dataset, bucket, key)
    except Exception as boto_exc:
        if not _s3_petrel_fallback_enabled():
            raise
        logger.warning(
            "boto3 failed for s3://%s/%s, falling back to Petrel: %s",
            bucket,
            key,
            boto_exc,
        )
        return _read_object_bytes_petrel(dataset, bucket, key)


def _is_image_key(key: str) -> bool:
    lower = key.lower()
    return any(lower.endswith(ext.lower()) for ext in IMG_EXTENSIONS)


class S3ImageFolder(Dataset):
    """
    ImageFolder-like dataset on S3 via Petrel SDK: root prefix must contain one subdirectory
    per class (e.g. .../CLS-LOC/train/<wnid>/*.JPEG).
    """

    def __init__(
        self,
        s3_path: str,
        endpoint_url: Optional[str] = None,
        transform: Optional[Callable] = None,
        region: str = "us-east-1",
        index: Optional[Dict[str, Any]] = None,
        petrel_conf: Optional[str] = None,
    ) -> None:
        self.transform = transform
        self.petrel_conf = petrel_conf
        self._client: Optional[Any] = None
        if index is not None:
            self.bucket = index["bucket"]
            self.root_prefix = index["root_prefix"]
            self.classes = index["classes"]
            self.class_to_idx = index["class_to_idx"]
            self.samples: List[Tuple[str, int]] = index["samples"]
        else:
            self.bucket, self.root_prefix = parse_s3_url(s3_path)
            client = _petrel_client(petrel_conf)
            self.classes, self.class_to_idx, self.samples = self._scan_tree(client)

    def __getstate__(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["_client"] = None
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _petrel_client(self.petrel_conf)
        return self._client

    def _scan_tree(self, client: Any) -> Tuple[List[str], Dict[str, int], List[Tuple[str, int]]]:
        prefix = self.root_prefix
        root_uri = _s3_uri(self.bucket, prefix)
        classes: List[str] = []
        for item in client.list(root_uri):
            if not item.endswith("/"):
                continue
            name = item.rstrip("/").split("/")[-1]
            if name:
                classes.append(name)
        if not classes:
            raise RuntimeError(
                f"No class subfolders under {root_uri} "
                "(expected ImageFolder layout: root/<class_name>/images)."
            )
        classes.sort()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        samples: List[Tuple[str, int]] = []
        for cls in classes:
            cls_uri = _s3_uri(self.bucket, f"{prefix}{cls}/")
            for item in client.list(cls_uri):
                if item.endswith("/") or not _is_image_key(item):
                    continue
                samples.append((item, class_to_idx[cls]))
        if not samples:
            raise RuntimeError(f"No image objects found under {root_uri}")
        return classes, class_to_idx, samples

    @staticmethod
    def build_index(
        s3_path: str,
        endpoint_url: Optional[str] = None,
        region: str = "us-east-1",
        petrel_conf: Optional[str] = None,
    ) -> Dict[str, Any]:
        bucket, root_prefix = parse_s3_url(s3_path)
        client = _petrel_client(petrel_conf)
        tmp = S3ImageFolder.__new__(S3ImageFolder)
        tmp.bucket = bucket
        tmp.root_prefix = root_prefix
        classes, class_to_idx, samples = tmp._scan_tree(client)
        return {
            "bucket": bucket,
            "root_prefix": root_prefix,
            "classes": classes,
            "class_to_idx": class_to_idx,
            "samples": samples,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        key, target = self.samples[index]
        data = _read_object_bytes(self, self.bucket, key)
        sample = Image.open(io.BytesIO(data)).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class LocalLabelFileDataset(Dataset):
    """Label-file dataset on a local directory (same line format as S3LabelFileDataset)."""

    def __init__(
        self,
        root: str,
        label_file: str = "",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root.rstrip("/")
        self.transform = transform
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Local data root not found: {self.root}")
        if not os.path.isfile(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        self.samples: List[Tuple[str, int]] = []
        with open(label_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.rsplit(maxsplit=1)
                self.samples.append((rel_path, int(label)))
        if not self.samples:
            raise RuntimeError(f"No samples found in label file: {label_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image_path = os.path.join(self.root, rel_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class S3LabelFileDataset(Dataset):
    """
    Read samples from a label file with lines formatted as:
        relative/path/to/image.jpg <label>
    Images are fetched from S3 under `s3_path` prefix via Petrel SDK.
    """

    def __init__(
        self,
        s3_path: str,
        endpoint_url: Optional[str] = None,
        label_file: str = "",
        transform: Optional[Callable] = None,
        region: str = "us-east-1",
        petrel_conf: Optional[str] = None,
    ) -> None:
        self.transform = transform
        self.petrel_conf = petrel_conf
        self.bucket, self.root_prefix = parse_s3_url(s3_path)
        self._client: Optional[Any] = None

        if not os.path.isfile(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        self.samples: List[Tuple[str, int]] = []
        with open(label_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.rsplit(maxsplit=1)
                key = rel_path if rel_path.startswith(self.root_prefix) else f"{self.root_prefix}{rel_path}"
                self.samples.append((key, int(label)))

        if not self.samples:
            raise RuntimeError(f"No samples found in label file: {label_file}")

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _petrel_client(self.petrel_conf)
        return self._client

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        key, label = self.samples[index]
        data = _read_object_bytes(self, self.bucket, key)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
