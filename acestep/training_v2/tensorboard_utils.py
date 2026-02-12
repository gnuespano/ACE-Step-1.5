"""
Extended TensorBoard Logging for ACE-Step Training V2

Provides helpers for:
    - Per-layer gradient norms
    - Learning rate tracking
    - Loss curves
    - Estimation score logging
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None  # type: ignore[misc,assignment]


def is_tensorboard_available() -> bool:
    """
    Check whether TensorBoard's SummaryWriter is available for import.
    
    Returns:
        `True` if `torch.utils.tensorboard.SummaryWriter` is importable, `false` otherwise.
    """
    return _TB_AVAILABLE


class TrainingLogger:
    """Wrapper around ``SummaryWriter`` with training-specific helpers.

    If TensorBoard is not installed, all methods are silent no-ops.
    """

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        """
        Initialize the TrainingLogger, creating a TensorBoard SummaryWriter when available.
        
        Creates the log directory if necessary and instantiates an internal SummaryWriter writing to `log_dir` when `enabled` is true and TensorBoard is importable; otherwise leaves the logger disabled and emits a warning if logging was requested but TensorBoard is unavailable.
        
        Parameters:
            log_dir (str | Path): Filesystem path where TensorBoard event files will be written.
            enabled (bool): Whether to enable TensorBoard logging; has effect only if TensorBoard is installed.
        """
        self._writer: Optional[Any] = None
        self._enabled = enabled and _TB_AVAILABLE
        if self._enabled:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))
            logger.info("[OK] TensorBoard logger initialised at %s", log_dir)
        else:
            if enabled and not _TB_AVAILABLE:
                logger.warning(
                    "[WARN] tensorboard not installed -- logging disabled. "
                    "Install with: pip install tensorboard"
                )

    # ------------------------------------------------------------------
    # Basic scalars
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value to the configured TensorBoard writer under a tag at a specific training step.
        
        If TensorBoard is not available or the writer is disabled, this method is a no-op.
        
        Parameters:
        	tag (str): Tag name used in TensorBoard for this scalar (e.g., "train/loss").
        	value (float): Numeric scalar value to record.
        	step (int): Global training step associated with this value.
        """
        if self._writer is not None:
            self._writer.add_scalar(tag, value, global_step=step)

    def log_loss(self, loss: float, step: int) -> None:
        """
        Log the training loss to the configured TensorBoard writer.
        
        Parameters:
        	loss (float): Training loss value to record.
        	step (int): Global training step index at which to log the loss.
        """
        self.log_scalar("train/loss", loss, step)

    def log_lr(self, lr: float, step: int) -> None:
        """
        Log the current learning rate under the "train/lr" tag.
        
        Parameters:
            lr (float): Learning rate value to record.
            step (int): Training step index associated with the value.
        """
        self.log_scalar("train/lr", lr, step)

    def log_epoch_loss(self, loss: float, epoch: int) -> None:
        """
        Record the epoch-end training loss to TensorBoard under the tag "train/epoch_loss".
        
        Parameters:
        	loss (float): The training loss value for the completed epoch.
        	epoch (int): The epoch index used as the logging step.
        """
        self.log_scalar("train/epoch_loss", loss, epoch)

    def log_grad_norm(self, norm: float, step: int) -> None:
        """
        Log the training gradient norm under the tag "train/grad_norm".
        
        Parameters:
            norm (float): L2 norm of gradients to record.
            step (int): Global step index associated with the measurement.
        """
        self.log_scalar("train/grad_norm", norm, step)

    # ------------------------------------------------------------------
    # Per-layer gradient norms (heavy)
    # ------------------------------------------------------------------

    def log_per_layer_grad_norms(
        self,
        model: nn.Module,
        step: int,
        prefix: str = "grad_norm",
    ) -> Dict[str, float]:
        """
        Compute and log the L2 gradient norm for each trainable parameter that has a gradient.
        
        Only parameters with requires_grad=True and a non-None .grad are included. Each computed norm is logged under the tag "{prefix}/{param_name}" when a writer is available.
        
        Parameters:
            model (nn.Module): Module whose named parameters will be inspected.
            step (int): Global step value used when logging scalars.
            prefix (str): Tag prefix to use for each parameter (default "grad_norm").
        
        Returns:
            Dict[str, float]: Mapping from "{prefix}/{param_name}" to the computed L2 gradient norm.
        """
        norms: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm_val = param.grad.data.float().norm(2).item()
                tag = f"{prefix}/{name}"
                norms[tag] = norm_val
                if self._writer is not None:
                    self._writer.add_scalar(tag, norm_val, global_step=step)
        return norms

    # ------------------------------------------------------------------
    # Estimation scores
    # ------------------------------------------------------------------

    def log_estimation_scores(
        self,
        scores: Dict[str, float],
        step: int = 0,
        prefix: str = "estimation",
    ) -> None:
        """
        Log per-module estimation scores to TensorBoard under a common prefix.
        
        Parameters:
            scores (Dict[str, float]): Mapping from module name to its estimation score.
            step (int): Global step at which to record the scalars.
            prefix (str): Tag prefix for each scalar; each score is logged under "{prefix}/{module_name}".
        """
        for module_name, score in scores.items():
            tag = f"{prefix}/{module_name}"
            self.log_scalar(tag, score, step)

    # ------------------------------------------------------------------
    # Histogram helpers
    # ------------------------------------------------------------------

    def log_param_histogram(
        self,
        model: nn.Module,
        step: int,
        prefix: str = "params",
    ) -> None:
        """
        Log histograms of all trainable parameter tensors in a module.
        
        Parameters:
            model (nn.Module): Module whose named parameters will be recorded.
            step (int): Global step value to associate with the logged histograms.
            prefix (str): Tag prefix used for each histogram; each parameter is logged under "{prefix}/{parameter_name}".
        """
        if self._writer is None:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._writer.add_histogram(
                    f"{prefix}/{name}", param.data.float().cpu(), global_step=step
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """
        Flush pending events to the underlying SummaryWriter if present.
        
        This is a no-op when the internal writer is not initialized.
        """
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """
        Close the internal TensorBoard writer and release its resources.
        
        If a writer has been created, closes it and clears the internal reference. Safe to call when no writer is present (no-op).
        """
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "TrainingLogger":
        """
        Enter the context manager and yield this TrainingLogger instance.
        
        Returns:
            self (TrainingLogger): The active logger to use within the context block.
        """
        return self

    def __exit__(self, *exc: Any) -> None:
        """
        Close the TrainingLogger when exiting a context manager.
        
        Parameters:
            *exc (Any): Exception information provided by the context manager protocol (exc_type, exc_value, traceback). These values are ignored.
        """
        self.close()