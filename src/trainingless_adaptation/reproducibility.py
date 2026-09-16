from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import site
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False


def sample_seed(filename: str, order: int, snr_db: float, global_seed: int) -> int:
    payload = f"{global_seed}|{filename}|{order}|{snr_db:g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def write_environment_manifest(destination: Path, extra: dict[str, Any] | None = None) -> None:
    conda_candidates = []
    for package_path in site.getsitepackages():
        path = Path(package_path)
        for distance, parent in enumerate((path, *path.parents)):
            if (parent / "conda-meta").is_dir():
                conda_candidates.append((distance, parent))
                break
    conda_prefix = min(conda_candidates, default=(0, None), key=lambda item: item[0])[1]
    conda_executable = None
    if conda_prefix is not None:
        for parent in (conda_prefix, *conda_prefix.parents):
            candidate = parent / "bin" / "conda"
            if candidate.exists():
                conda_executable = candidate
                break
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_paths": site.getsitepackages(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "pip_freeze": command_output([os.sys.executable, "-m", "pip", "freeze"]),
        "conda_prefix_listed": str(conda_prefix) if conda_prefix is not None else None,
        "conda_list": command_output([str(conda_executable), "list", "-p", str(conda_prefix)])
        if conda_executable is not None
        else "unavailable: no parent Conda prefix found",
    }
    if extra:
        payload.update(extra)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
