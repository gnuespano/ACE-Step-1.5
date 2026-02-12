"""
Cached GPU / VRAM monitor for the live training display.

Wraps ``gpu_utils`` with a time-based cache to avoid hammering the GPU
driver on every render tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class VRAMSnapshot:
    """A point-in-time GPU memory reading."""

    used_mb: float = 0.0
    total_mb: float = 0.0
    name: str = "unknown"
    timestamp: float = 0.0

    @property
    def used_gb(self) -> float:
        """
        Used GPU memory in gibibytes (GiB).
        
        Returns:
            float: Used memory in GiB.
        """
        return self.used_mb / 1024.0

    @property
    def total_gb(self) -> float:
        """
        Total VRAM expressed in gibibytes (GiB).
        
        Returns:
            Total VRAM in GiB (computed from `total_mb`).
        """
        return self.total_mb / 1024.0

    @property
    def percent(self) -> float:
        """
        Compute used VRAM as a percentage of total VRAM.
        
        Returns:
            percent (float): The percent of VRAM used (used_mb / total_mb * 100). Returns 0.0 if total_mb is less than or equal to 0.
        """
        if self.total_mb <= 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100.0

    @property
    def free_mb(self) -> float:
        """
        Return the amount of free VRAM in mebibytes.
        
        Returns:
            free_mb (float): Free VRAM in MiB; returns 0.0 if used_mb is greater than total_mb.
        """
        return max(0.0, self.total_mb - self.used_mb)


class GPUMonitor:
    """Cached GPU VRAM monitor.

    Queries the device at most every ``interval`` seconds.  Returns the
    last snapshot otherwise.

    Args:
        device: Torch device string (``cuda:0``, ``mps``, ``cpu``, ...).
        interval: Minimum seconds between actual GPU queries.
    """

    def __init__(self, device: str = "cuda:0", interval: float = 5.0) -> None:
        """
        Initialize the GPUMonitor with a device identifier and a snapshot cache interval.
        
        Parameters:
            device: Device string identifying the target device (e.g., "cuda:0" or "cpu"). The portion before ":" is used as the device type and determines whether CUDA-based monitoring is attempted.
            interval: Minimum number of seconds between fresh GPU queries; snapshots requested sooner than this interval return the cached snapshot.
        
        Notes:
            The initializer sets internal state (including availability based on device type) and calls _init_static() to cache static device information such as name and total VRAM.
        """
        self._device = device
        self._device_type = device.split(":")[0]
        self._interval = interval
        self._last: Optional[VRAMSnapshot] = None
        self._available = self._device_type == "cuda"
        self._name: str = ""
        self._total_mb: float = 0.0
        self._init_static()

    # ---- static (one-time) queries -----------------------------------------

    def _init_static(self) -> None:
        """Cache device name and total VRAM (these don't change)."""
        if not self._available:
            return
        try:
            import torch

            idx = self._cuda_idx()
            self._name = torch.cuda.get_device_name(idx)
            self._total_mb = (
                torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2)
            )
        except Exception:
            self._available = False

    def _cuda_idx(self) -> int:
        """
        Parse the CUDA device index from the monitor's device string.
        
        Returns:
            int: The CUDA device index parsed from the portion after ':' in the device string, or 0 if no index is present.
        """
        if ":" in self._device:
            return int(self._device.split(":")[1])
        return 0

    # ---- public API ---------------------------------------------------------

    @property
    def available(self) -> bool:
        """
        Indicates whether VRAM monitoring is available for the configured device.
        
        Returns:
            `true` if VRAM monitoring is available for the configured device, `false` otherwise.
        """
        return self._available

    def snapshot(self) -> VRAMSnapshot:
        """
        Get a cached VRAMSnapshot, refreshing the snapshot only when no cached value exists or the configured interval has elapsed.
        
        Returns:
            VRAMSnapshot: Snapshot with `timestamp` always set. When GPU monitoring is available and successful, `used_mb`, `total_mb`, and `name` are populated; if measurement fails or monitoring is unavailable, only `timestamp` (and cached `total_mb`/`name` when known) are provided.
        """
        now = time.monotonic()

        if self._last is not None and (now - self._last.timestamp) < self._interval:
            return self._last

        if not self._available:
            snap = VRAMSnapshot(timestamp=now)
            self._last = snap
            return snap

        try:
            import torch

            idx = self._cuda_idx()
            torch.cuda.synchronize(idx)
            reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
            snap = VRAMSnapshot(
                used_mb=reserved,
                total_mb=self._total_mb,
                name=self._name,
                timestamp=now,
            )
        except Exception:
            snap = VRAMSnapshot(
                total_mb=self._total_mb,
                name=self._name,
                timestamp=now,
            )

        self._last = snap
        return snap

    def peak_mb(self) -> float:
        """
        Get peak allocated GPU VRAM for the configured CUDA device in mebibytes.
        
        Returns:
            peak_mb (float): Peak allocated VRAM in MiB, or 0.0 if CUDA is unavailable or the value cannot be retrieved.
        """
        if not self._available:
            return 0.0
        try:
            import torch
            return torch.cuda.max_memory_allocated(self._cuda_idx()) / (1024 ** 2)
        except Exception:
            return 0.0