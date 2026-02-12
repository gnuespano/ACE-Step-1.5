"""
GPU Detection, VRAM Query, and Estimation Budget Calculation

Provides:
    detect_gpu()            -- auto-detect best available accelerator
    get_available_vram_mb() -- query free VRAM on the selected device
    estimate_batch_budget() -- compute how many estimation batches fit in VRAM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Describes the detected accelerator."""

    device: str
    """Torch device string, e.g. 'cuda:0', 'mps', 'xpu', 'cpu'."""

    device_type: str
    """Canonical type: 'cuda', 'mps', 'xpu', 'cpu'."""

    name: str
    """Human-readable device name."""

    vram_total_mb: Optional[float] = None
    """Total device memory in MiB (None for CPU / unknown)."""

    vram_free_mb: Optional[float] = None
    """Free device memory in MiB (None for CPU / unknown)."""

    precision: str = "fp32"
    """Recommended precision for this device."""

    def __str__(self) -> str:
        """
        Return a readable string representation of the GPUInfo.
        
        The representation is formatted as "GPUInfo(device=..., name=..., vram_total=...MiB, vram_free=...MiB, precision=...)", omitting the vram_total and vram_free fields when their values are unknown.
        
        Returns:
            A formatted string summarizing the GPUInfo instance.
        """
        parts = [f"device={self.device}", f"name={self.name}"]
        if self.vram_total_mb is not None:
            parts.append(f"vram_total={self.vram_total_mb:.0f}MiB")
        if self.vram_free_mb is not None:
            parts.append(f"vram_free={self.vram_free_mb:.0f}MiB")
        parts.append(f"precision={self.precision}")
        return "GPUInfo(" + ", ".join(parts) + ")"


def detect_gpu(requested_device: str = "auto", requested_precision: str = "auto") -> GPUInfo:
    """
    Detect the best available accelerator and return its metadata as a GPUInfo.
    
    Parameters:
        requested_device: Preferred device selection. One of 'auto', 'cuda', 'cuda:N', 'mps', 'xpu', or 'cpu'. When 'auto', preference order is CUDA, then MPS, then XPU, then CPU.
        requested_precision: Preferred computation precision. One of 'auto', 'bf16', 'fp16', or 'fp32'. When 'auto', selects 'bf16' for CUDA/XPU, 'fp16' for MPS, and 'fp32' for CPU.
    
    Returns:
        GPUInfo: Dataclass containing device string, canonical device_type, human-readable name, total VRAM (MB) or None, free VRAM (MB) or None, and recommended precision.
    """
    device_type: str
    device_str: str
    name: str
    vram_total: Optional[float] = None
    vram_free: Optional[float] = None

    if requested_device == "auto":
        if torch.cuda.is_available():
            device_str = "cuda:0"
            device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
            device_type = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_str = "xpu:0"
            device_type = "xpu"
        else:
            device_str = "cpu"
            device_type = "cpu"
    else:
        device_str = requested_device
        device_type = requested_device.split(":")[0]

    # Resolve name + VRAM
    if device_type == "cuda":
        idx = 0
        if ":" in device_str:
            idx = int(device_str.split(":")[1])
        name = torch.cuda.get_device_name(idx)
        vram_total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2)
        vram_free = _cuda_free_mb(idx)
    elif device_type == "mps":
        name = "Apple MPS"
    elif device_type == "xpu":
        idx = 0
        if ":" in device_str:
            idx = int(device_str.split(":")[1])
        name = f"Intel XPU:{idx}"
        if hasattr(torch.xpu, "get_device_properties"):
            props = torch.xpu.get_device_properties(idx)
            vram_total = getattr(props, "total_memory", 0) / (1024 ** 2)
    else:
        name = "CPU"

    # Resolve precision
    if requested_precision == "auto":
        if device_type in ("cuda", "xpu"):
            precision = "bf16"
        elif device_type == "mps":
            precision = "fp16"
        else:
            precision = "fp32"
    else:
        precision = requested_precision

    return GPUInfo(
        device=device_str,
        device_type=device_type,
        name=name,
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        precision=precision,
    )


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def _cuda_free_mb(idx: int = 0) -> float:
    """Return free CUDA memory in MiB for device *idx*."""
    torch.cuda.synchronize(idx)
    free, _total = torch.cuda.mem_get_info(idx)
    return free / (1024 ** 2)


def get_available_vram_mb(device: str = "auto") -> Optional[float]:
    """
    Get available free VRAM for the specified device.
    
    Parameters:
    	device (str): Device to query, e.g. "auto" (auto-detect), "cuda:0", "mps", "cpu", or other device string.
    
    Returns:
    	free_vram_mb (float | None): Free VRAM in megabytes, or `None` if VRAM cannot be determined (for example on CPU or when information is unavailable).
    """
    info = detect_gpu(requested_device=device)
    return info.vram_free_mb


# ---------------------------------------------------------------------------
# Estimation batch budget
# ---------------------------------------------------------------------------

# Rough per-batch memory estimate for a single forward+backward pass through
# the full DiT decoder (24 layers, hidden_size=2048) with all parameters
# requiring gradients.  This is a *very* conservative upper bound.
_BYTES_PER_BATCH_ESTIMATE_BF16: float = 1200.0  # ~1.2 GiB per sample


def get_gpu_info(device: str = "auto") -> dict:
    """
    Provide a flat dictionary of detected GPU information suitable for UI widgets.
    
    Parameters:
        device (str): Device selector string (e.g., "auto", "cuda:0", "mps", "cpu"); "auto" selects the best available accelerator.
    
    Returns:
        dict: Mapping with the following keys:
            - name (str): Human-readable device name or "Unknown" on failure.
            - vram_used_gb (float): Estimated used VRAM in GiB (0 if unknown).
            - vram_total_gb (float): Total VRAM in GiB (0 if unknown).
            - utilization (int): Placeholder GPU utilization percentage (0).
            - temperature (int): Placeholder GPU temperature (0).
            - power (int): Placeholder GPU power consumption (0).
    """
    try:
        info = detect_gpu(requested_device=device)
        total_mb = info.vram_total_mb or 0
        free_mb = info.vram_free_mb or 0
        used_mb = max(0, total_mb - free_mb)
        return {
            "name": info.name,
            "vram_used_gb": used_mb / 1024,
            "vram_total_gb": total_mb / 1024,
            "utilization": 0,  # nvidia-smi would be needed for live util
            "temperature": 0,
            "power": 0,
        }
    except Exception:
        return {
            "name": "Unknown",
            "vram_used_gb": 0,
            "vram_total_gb": 0,
            "utilization": 0,
            "temperature": 0,
            "power": 0,
        }


def estimate_batch_budget(
    device: str = "auto",
    safety_factor: float = 0.8,
    min_batches: int = 4,
    max_batches: int = 64,
) -> int:
    """
    Estimate how many estimation batches fit in the available VRAM for a device.
    
    This function queries free VRAM for the specified device, applies a safety factor, and reserves
    approximately 4096 MiB for model weights before computing how many estimation batches (using an
    internal per-batch memory estimate for bf16) will fit. The result is clamped to the provided
    minimum and maximum batch bounds.
    
    Parameters:
        device (str): Device identifier or "auto" to auto-detect the best available accelerator.
        safety_factor (float): Fraction of free VRAM to consider usable (value between 0 and 1).
        min_batches (int): Minimum number of batches to return if VRAM is very limited or unknown.
        max_batches (int): Maximum number of batches to return.
    
    Returns:
        int: Number of estimation batches, clamped to the range [min_batches, max_batches].
    """
    free_mb = get_available_vram_mb(device)
    if free_mb is None:
        logger.info("[INFO] VRAM unknown -- using minimum batch budget of %d", min_batches)
        return min_batches

    usable_mb = free_mb * safety_factor
    # Subtract ~4 GiB for the model weights themselves
    usable_mb = max(0.0, usable_mb - 4096.0)
    n_batches = int(usable_mb / _BYTES_PER_BATCH_ESTIMATE_BF16)
    n_batches = max(min_batches, min(n_batches, max_batches))

    logger.info(
        "[INFO] Estimation budget: %d batches (%.0f MiB free, %.0f MiB usable)",
        n_batches,
        free_mb,
        usable_mb,
    )
    return n_batches