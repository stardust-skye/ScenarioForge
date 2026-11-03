"""Random seed setup and experiment environment capture."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def initialize_seed(seed: int, *, mode: str = "strict") -> torch.Generator:
    """Seed Python, NumPy, PyTorch, CUDA, and a DataLoader generator."""
    if mode not in ("strict", "best_effort"):
        raise ValueError("mode must be 'strict' or 'best_effort'")
    if mode == "strict":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in (":4096:8", ":16:8"):
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "Strict CUDA reproducibility requires CUBLAS_WORKSPACE_CONFIG="
                    "':4096:8' or ':16:8' before CUDA initialization"
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=mode == "best_effort")
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state using values supported by safe ``torch.load``."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore Python, NumPy, CPU PyTorch, and available CUDA RNG states."""
    validate_rng_state(state)
    random.setstate(_nested_tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state((
        str(numpy_state["algorithm"]),
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Checkpoint CUDA RNG state count does not match available CUDA devices"
            )
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def validate_rng_state(state: Mapping[str, Any]) -> None:
    """Validate a checkpoint RNG payload before mutating any process state."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not required <= state.keys():
        raise ValueError("Checkpoint RNG state is incomplete")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or not {
        "algorithm", "keys", "position", "has_gauss", "cached_gaussian"
    } <= numpy_state.keys():
        raise ValueError("Checkpoint NumPy RNG state is invalid")
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise ValueError("Checkpoint CPU PyTorch RNG state is invalid")
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list) or any(
        not isinstance(item, torch.Tensor) or item.dtype != torch.uint8
        for item in cuda_states
    ):
        raise ValueError("Checkpoint CUDA RNG state is invalid")
    if cuda_states and (
        not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()
    ):
        raise ValueError(
            "Checkpoint CUDA RNG state count does not match available CUDA devices"
        )


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(_nested_tuple(item) for item in value)
    return value


def seed_data_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the per-worker PyTorch seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def environment_metadata(device: torch.device) -> dict[str, Any]:
    """Record runtime versions and hardware used for the experiment."""
    packages = {}
    for package in ("scenarioforge", "torch", "torchvision", "torchmetrics", "faster-coco-eval", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    hardware: dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware["gpu_name"] = properties.name
        hardware["gpu_total_memory_bytes"] = properties.total_memory
    cudnn = getattr(torch.backends, "cudnn", None)
    hardware["cudnn_version"] = torch.backends.cudnn.version() if cudnn else None
    hardware["deterministic_settings"] = {
        "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": bool(cudnn.deterministic) if cudnn else None,
        "cudnn_benchmark": bool(cudnn.benchmark) if cudnn else None,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(cudnn.allow_tf32) if cudnn else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "hardware": hardware,
    }
