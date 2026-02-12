"""
Side-Step Optimizer & Scheduler Factories

Provides ``build_optimizer()`` and ``build_scheduler()`` so that
``trainer_fixed.py`` doesn't need to hard-code AdamW / CosineAnnealing.

Supported optimizers:
    adamw       -- torch.optim.AdamW (default, fused on CUDA)
    adamw8bit   -- bitsandbytes.optim.AdamW8bit (optional dep)
    adafactor   -- transformers.optimization.Adafactor
    prodigy     -- prodigyopt.Prodigy (optional dep, auto-tunes LR)

Supported schedulers:
    cosine              -- warmup + CosineAnnealingWarmRestarts
    linear              -- warmup + LinearLR decay to near-zero
    constant            -- warmup then flat LR
    constant_with_warmup -- alias for constant
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ConstantLR,
    LinearLR,
    SequentialLR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(
    params: Iterable,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device_type: str = "cuda",
) -> torch.optim.Optimizer:
    """
    Create an optimizer instance selected by name with sensible fallbacks when optional backends are unavailable.
    
    Supported optimizer_type values: "adamw" (default), "adamw8bit", "adafactor", "prodigy". If an optional implementation (bitsandbytes, transformers, or prodigyopt) is not installed, the function falls back to using AdamW.
    
    Parameters:
        params (Iterable): Iterable of model parameters to optimize.
        optimizer_type (str): Key selecting the optimizer implementation; case-insensitive.
        lr (float): Base learning rate to configure the optimizer.
        weight_decay (float): Weight decay (L2 penalty) to configure the optimizer.
        device_type (str): Device hint; when "cuda", AdamW is created with fused=True when available.
    
    Returns:
        torch.optim.Optimizer: An optimizer instance configured for the provided parameters.
    """
    optimizer_type = optimizer_type.lower().strip()

    if optimizer_type == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
            logger.info("[Side-Step] Using AdamW8bit optimizer (lower VRAM)")
            return AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            logger.warning(
                "[Side-Step] bitsandbytes not installed -- falling back to AdamW. "
                "Install with: pip install bitsandbytes>=0.45.0"
            )
            optimizer_type = "adamw"

    if optimizer_type == "adafactor":
        try:
            from transformers.optimization import Adafactor
            logger.info("[Side-Step] Using Adafactor optimizer (minimal state memory)")
            return Adafactor(
                params,
                lr=lr,
                weight_decay=weight_decay,
                scale_parameter=False,
                relative_step=False,
            )
        except ImportError:
            logger.warning(
                "[Side-Step] transformers not installed -- falling back to AdamW"
            )
            optimizer_type = "adamw"

    if optimizer_type == "prodigy":
        try:
            from prodigyopt import Prodigy
            logger.info(
                "[Side-Step] Using Prodigy optimizer (adaptive LR -- set LR=1.0 for best results)"
            )
            return Prodigy(
                params,
                lr=lr if lr != 1e-4 else 1.0,  # Default to 1.0 for Prodigy
                weight_decay=weight_decay,
            )
        except ImportError:
            logger.warning(
                "[Side-Step] prodigyopt not installed -- falling back to AdamW. "
                "Install with: pip install prodigyopt>=1.1.2"
            )
            optimizer_type = "adamw"

    # Default: AdamW
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if device_type == "cuda":
        kwargs["fused"] = True
    logger.info("[Side-Step] Using AdamW optimizer")
    return AdamW(params, **kwargs)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 1000,
    warmup_steps: int = 500,
    lr: float = 1e-4,
    optimizer_type: str = "adamw",
):
    """
    Builds a composite learning-rate scheduler with a linear warmup followed by a selectable main schedule.
    
    If `optimizer_type` is "prodigy", the scheduler is forced to "constant" because Prodigy manages learning rates internally. `warmup_steps` is clamped to at most `max(1, total_steps // 10)`. Supported `scheduler_type` values:
    - "constant" or "constant_with_warmup": constant LR after warmup (ConstantLR, factor=1.0, total_iters=total_steps).
    - "linear": linear decay from factor 1.0 to 0.01 over (total_steps - warmup_steps).
    - "cosine" (default): cosine annealing with restarts using CosineAnnealingWarmRestarts with T_0 = max(1, total_steps - warmup_steps) and eta_min = lr * 0.01.
    
    Returns:
        A torch.optim.lr_scheduler.SequentialLR that applies a LinearLR warmup (scaling from 0.1 to 1.0 over `warmup_steps`) followed by the selected main scheduler, with the transition milestone at `warmup_steps`.
    """
    scheduler_type = scheduler_type.lower().strip()

    # Prodigy handles its own LR -- force constant
    if optimizer_type == "prodigy" and scheduler_type not in ("constant", "constant_with_warmup"):
        logger.info(
            "[Side-Step] Prodigy optimizer detected -- overriding scheduler to 'constant' "
            "(Prodigy adapts LR internally)"
        )
        scheduler_type = "constant"

    # Clamp warmup to avoid exceeding total
    warmup_steps = min(warmup_steps, max(1, total_steps // 10))

    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    if scheduler_type in ("constant", "constant_with_warmup"):
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
    elif scheduler_type == "linear":
        remaining = max(1, total_steps - warmup_steps)
        main_sched = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=remaining,
        )
    else:
        # cosine (default)
        main_sched = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - warmup_steps),
            T_mult=1,
            eta_min=lr * 0.01,
        )

    return SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup_steps])