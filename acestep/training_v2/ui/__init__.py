"""
ACE-Step Training V2 -- Rich UI Layer

Provides a shared ``Console`` instance and a ``RICH_AVAILABLE`` flag so that
every UI module can degrade gracefully when Rich is not installed.

Exports
-------
console : Console | None
    Shared Rich console (``None`` when Rich is missing).
RICH_AVAILABLE : bool
    ``True`` when ``rich>=13`` is importable.
plain_mode : bool
    Module-level flag toggled by ``--plain``.  When ``True`` *or* stdout is
    not a TTY, all UI helpers fall back to plain ``print()`` output.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterator, Tuple

# ---- Rich availability check ------------------------------------------------

RICH_AVAILABLE: bool = False
console = None  # type: ignore[assignment]

try:
    from rich.console import Console as _Console

    console = _Console(stderr=True)  # UI goes to stderr so stdout stays clean
    RICH_AVAILABLE = True
except ImportError:
    pass

# ---- Plain-mode flag (set via --plain CLI arg) ------------------------------

plain_mode: bool = False


def set_plain_mode(value: bool) -> None:
    """
    Set the global plain-mode flag to force plain-text UI output.
    
    Parameters:
        value (bool): True to enable plain-mode (use plain text output), False to disable it.
    """
    global plain_mode
    plain_mode = value


def is_rich_active() -> bool:
    """
    Determine whether Rich-backed console output should be used.
    
    Returns:
        True if Rich is available, plain mode is not enabled, and the configured console is a terminal; `False` otherwise.
    """
    if plain_mode or not RICH_AVAILABLE:
        return False
    if console is not None and not console.is_terminal:
        return False
    return True


def require_rich() -> None:
    """
    Ensure the Rich library is available; if not, write an installation hint to stderr and exit the process with status 1.
    
    If Rich is already available this function returns immediately. When Rich is missing it prints a short message suggesting `pip install rich` or using `--plain` and then calls `sys.exit(1)`.
    """
    if RICH_AVAILABLE:
        return
    print(
        "[FAIL] Rich is required for the pretty CLI.\n"
        "       Install it with:  pip install rich\n"
        "       Or use --plain for basic text output.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---- TrainingUpdate (backward-compatible structured yield) ------------------

@dataclass
class TrainingUpdate:
    """Structured object yielded by the trainer, backward-compatible with
    ``(step, loss, msg)`` tuple unpacking.

    Extra fields give the UI enough context to render a live dashboard
    without parsing message strings.
    """

    step: int
    loss: float
    msg: str
    kind: str = "info"
    """One of: info, step, epoch, checkpoint, complete, warn, fail."""
    epoch: int = 0
    max_epochs: int = 0
    lr: float = 0.0
    epoch_time: float = 0.0
    samples_per_sec: float = 0.0

    # -- backward compat: ``for step, loss, msg in trainer.train():`` --------
    def __iter__(self) -> Iterator[Tuple[int, float, str]]:  # type: ignore[override]
        """
        Provide backward-compatible iteration yielding the (step, loss, msg) tuple.
        
        Returns:
            iterator (Iterator[Tuple[int, float, str]]): An iterator over a three-tuple containing `step`, `loss`, and `msg`.
        """
        return iter((self.step, self.loss, self.msg))