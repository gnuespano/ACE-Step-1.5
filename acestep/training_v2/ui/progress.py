"""
Live training progress display using Rich.

Renders a live-updating dashboard that shows:
    - Epoch progress bar with ETA
    - Current metrics (loss, learning rate, speed)
    - GPU VRAM usage bar
    - Scrolling log of recent messages

Falls back to plain ``print(msg)`` when Rich is unavailable or stdout
is not a TTY.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator, Optional, Tuple, Union

from acestep.training_v2.ui import TrainingUpdate, console, is_rich_active
from acestep.training_v2.ui.gpu_monitor import GPUMonitor


# ---- Training statistics tracker --------------------------------------------

@dataclass
class TrainingStats:
    """Accumulates statistics during training for the live display and
    the post-training summary.
    """

    start_time: float = 0.0
    first_loss: float = 0.0
    best_loss: float = float("inf")
    last_loss: float = 0.0
    last_lr: float = 0.0
    _lr_seen: bool = False
    current_epoch: int = 0
    max_epochs: int = 0
    current_step: int = 0
    total_steps_estimate: int = 0
    steps_this_session: int = 0
    peak_vram_mb: float = 0.0
    last_epoch_time: float = 0.0
    _step_times: list = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        """
        Elapsed seconds since the recorded training start time.
        
        Returns:
            Elapsed time in seconds as a float; returns 0.0 if the start time has not been set.
        """
        if self.start_time <= 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def elapsed_str(self) -> str:
        """
        Formatted human-readable string representing total elapsed training time.
        
        Returns:
            A string with the elapsed duration formatted (hours/minutes/seconds) or "--" when no time is recorded.
        """
        return _fmt_duration(self.elapsed)

    @property
    def samples_per_sec(self) -> float:
        """
        Compute the recent processing speed in steps per second.
        
        Uses the recorded step timestamps to estimate speed; returns 0.0 if fewer than two timestamps are available or if the time span is non-positive.
        
        Returns:
            float: Steps per second based on the sliding window of recorded step times, or `0.0` when not computable.
        """
        if not self._step_times or len(self._step_times) < 2:
            return 0.0
        dt = self._step_times[-1] - self._step_times[0]
        if dt <= 0:
            return 0.0
        return (len(self._step_times) - 1) / dt

    @property
    def eta_seconds(self) -> float:
        """
        Estimate remaining training time in seconds based on completed epochs.
        
        Returns:
            Estimated remaining time in seconds; 0.0 if the estimate cannot be computed
            (for example when max_epochs or current_epoch is not set or elapsed time is zero).
        """
        if self.max_epochs <= 0 or self.current_epoch <= 0:
            return 0.0
        elapsed = self.elapsed
        if elapsed <= 0:
            return 0.0
        progress = self.current_epoch / self.max_epochs
        if progress <= 0:
            return 0.0
        return elapsed * (1.0 / progress - 1.0)

    @property
    def eta_str(self) -> str:
        """
        Return a formatted ETA string for the remaining training time.
        
        Returns:
            str: Human-readable ETA (e.g., "1h 2m", "3m 5s"), or "--" if an ETA cannot be estimated.
        """
        eta = self.eta_seconds
        if eta <= 0:
            return "--"
        return _fmt_duration(eta)

    def record_step(self) -> None:
        """
        Record the current time for a completed training step.
        
        Appends a timestamp to the internal step-time buffer and trims it to keep only the most recent 50 entries, which are used to estimate steps per second.
        """
        now = time.time()
        self._step_times.append(now)
        # Keep a sliding window of 50 timestamps for speed calculation
        if len(self._step_times) > 50:
            self._step_times = self._step_times[-50:]


def _fmt_duration(seconds: float) -> str:
    """Format seconds to ``1h 23m 45s`` or ``12m 34s`` or ``45s``."""
    if seconds < 0:
        return "--"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---- Rich live display builder ----------------------------------------------

def _build_display(
    stats: TrainingStats,
    gpu: GPUMonitor,
    recent_msgs: list,
) -> "Rich renderable":
    """
    Create the Rich renderable used for a single Live refresh of the training dashboard.
    
    Parameters:
        stats (TrainingStats): Aggregated training statistics used to populate progress, metrics, and timing.
        gpu (GPUMonitor): GPU monitor used to render VRAM usage when available.
        recent_msgs (list): Recent log messages shown in the dashboard (most recent five are displayed).
    
    Returns:
        A Rich renderable (Panel) containing the epoch progress bar, metrics table, VRAM usage line, and recent log text.
    """
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    # -- Epoch progress -------------------------------------------------------
    epoch_pct = 0.0
    if stats.max_epochs > 0:
        epoch_pct = stats.current_epoch / stats.max_epochs
    progress_bar = ProgressBar(total=100, completed=int(epoch_pct * 100), width=40)

    epoch_line = Text()
    epoch_line.append("  Epoch ", style="dim")
    epoch_line.append(f"{stats.current_epoch}", style="bold")
    epoch_line.append(f" / {stats.max_epochs}  ", style="dim")
    epoch_line.append_text(Text.from_markup(f"  Step {stats.current_step}"))
    epoch_line.append(f"  |  ETA {stats.eta_str}", style="dim")

    # -- Metrics table --------------------------------------------------------
    metrics = Table(show_header=False, show_edge=False, pad_edge=False, box=None, expand=True)
    metrics.add_column("key", style="dim", ratio=1)
    metrics.add_column("val", ratio=1)
    metrics.add_column("key2", style="dim", ratio=1)
    metrics.add_column("val2", ratio=1)

    # Loss formatting: color-code direction
    loss_str = f"{stats.last_loss:.4f}" if stats.last_loss > 0 else "--"
    best_str = f"{stats.best_loss:.4f}" if stats.best_loss < float("inf") else "--"
    lr_str = f"{stats.last_lr:.2e}" if stats._lr_seen else "--"
    speed_str = f"{stats.samples_per_sec:.1f} steps/s" if stats.samples_per_sec > 0 else "--"

    metrics.add_row("Loss", f"[bold]{loss_str}[/]", "Best", f"[green]{best_str}[/]")
    metrics.add_row("LR", lr_str, "Speed", speed_str)
    metrics.add_row("Elapsed", stats.elapsed_str, "Epoch time", f"{stats.last_epoch_time:.1f}s" if stats.last_epoch_time > 0 else "--")

    # -- VRAM bar -------------------------------------------------------------
    if gpu.available:
        snap = gpu.snapshot()
        pct = snap.percent
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar_color = "green" if pct < 70 else ("yellow" if pct < 90 else "red")
        bar = f"[{bar_color}]{'#' * filled}[/][dim]{'-' * (bar_width - filled)}[/]"
        vram_line = (
            f"  VRAM {bar}  "
            f"{snap.used_gb:.1f} / {snap.total_gb:.1f} GiB  "
            f"[dim]({pct:.0f}%)[/]"
        )
    else:
        vram_line = "  [dim]VRAM monitoring not available[/]"

    # -- Recent log -----------------------------------------------------------
    log_text = Text()
    for msg in recent_msgs[-5:]:
        if msg.startswith("[OK]"):
            log_text.append(f"  {msg}\n", style="green")
        elif msg.startswith("[WARN]"):
            log_text.append(f"  {msg}\n", style="yellow")
        elif msg.startswith("[FAIL]"):
            log_text.append(f"  {msg}\n", style="red")
        elif msg.startswith("[INFO]"):
            log_text.append(f"  {msg}\n", style="blue")
        else:
            log_text.append(f"  {msg}\n", style="dim")

    # -- Assemble panel -------------------------------------------------------
    from rich.console import Group

    parts = [
        epoch_line,
        Text(""),
        Text.from_markup(f"  {progress_bar}  [dim]{epoch_pct * 100:.0f}%[/]"),
        Text(""),
        metrics,
        Text(""),
        Text.from_markup(vram_line),
        Text(""),
        log_text,
    ]

    return Panel(
        Group(*parts),
        title="[bold]Side-Step Training Progress[/]",
        border_style="green",
        padding=(0, 1),
    )


# ---- Main entry point -------------------------------------------------------

def track_training(
    training_iter: Iterator[Union[Tuple[int, float, str], TrainingUpdate]],
    max_epochs: int,
    device: str = "cuda:0",
    refresh_per_second: int = 2,
) -> TrainingStats:
    """Consume training yields and display live progress.

    Args:
        training_iter: Generator yielding ``(step, loss, msg)`` or
            ``TrainingUpdate`` objects.
        max_epochs: Total number of epochs (for progress bar).
        device: Device string for GPU monitoring.
        refresh_per_second: Rich Live refresh rate.

    Returns:
        Final ``TrainingStats`` for the summary display.
    """
    stats = TrainingStats(start_time=time.time(), max_epochs=max_epochs)
    gpu = GPUMonitor(device=device, interval=3.0)
    recent_msgs: list[str] = []

    if is_rich_active() and console is not None:
        return _track_rich(training_iter, stats, gpu, recent_msgs, refresh_per_second)
    else:
        return _track_plain(training_iter, stats, gpu, recent_msgs)


def _track_rich(
    training_iter: Iterator,
    stats: TrainingStats,
    gpu: GPUMonitor,
    recent_msgs: list,
    refresh_per_second: int,
) -> TrainingStats:
    """
    Render a live training dashboard with Rich while consuming updates from the training iterator.
    
    This function drives a Rich Live display that is refreshed at the given rate. It accepts iterator items that are either TrainingUpdate objects or (step, loss, msg) tuples, updates the provided TrainingStats in-place for each item, appends messages to recent_msgs (capped at 20), and refreshes the display. When the iterator completes, the GPU monitor's peak VRAM is recorded on the returned stats.
    
    Parameters:
        training_iter: Iterator that yields either TrainingUpdate instances or (step, loss, msg) tuples.
        stats: TrainingStats object to update in-place with progress, timing, and metric information.
        gpu: GPUMonitor used to read and record VRAM usage.
        recent_msgs: List used to accumulate recent messages shown in the dashboard (mutated in-place).
        refresh_per_second: Number of display refreshes per second.
    
    Returns:
        stats (TrainingStats): The final TrainingStats populated during tracking, with peak_vram_mb updated.
    """
    from rich.live import Live

    assert console is not None

    with Live(
        _build_display(stats, gpu, recent_msgs),
        console=console,
        refresh_per_second=refresh_per_second,
        transient=True,  # Clear the live display when done
    ) as live:
        for update in training_iter:
            # Unpack (works for both tuples and TrainingUpdate)
            if isinstance(update, TrainingUpdate):
                step, loss, msg = update.step, update.loss, update.msg
                _process_structured(update, stats)
            else:
                step, loss, msg = update
                _process_tuple(step, loss, msg, stats)

            recent_msgs.append(msg)
            if len(recent_msgs) > 20:
                recent_msgs.pop(0)

            live.update(_build_display(stats, gpu, recent_msgs))

    # Record peak VRAM
    stats.peak_vram_mb = gpu.peak_mb()
    return stats


def _track_plain(
    training_iter: Iterator,
    stats: TrainingStats,
    gpu: GPUMonitor,
    recent_msgs: list,
) -> TrainingStats:
    """
    Display plain-text training progress: consume updates from `training_iter`, print each update's message, and update `stats`.
    
    Parameters:
        training_iter (Iterator): Yields either `TrainingUpdate` objects or `(step, loss, msg)` tuples.
        stats (TrainingStats): Mutable training statistics object that will be updated and returned.
        gpu (GPUMonitor): GPU monitor used to record peak VRAM usage.
    
    Returns:
        TrainingStats: The updated `stats` object with `peak_vram_mb` set from `gpu`.
    """
    for update in training_iter:
        if isinstance(update, TrainingUpdate):
            step, loss, msg = update.step, update.loss, update.msg
            _process_structured(update, stats)
        else:
            step, loss, msg = update
            _process_tuple(step, loss, msg, stats)

        print(msg)

    stats.peak_vram_mb = gpu.peak_mb()
    return stats


# ---- Update processing helpers ----------------------------------------------

def _process_structured(update: TrainingUpdate, stats: TrainingStats) -> None:
    """
    Update the given TrainingStats in place using values from a TrainingUpdate.
    
    Updates current step, loss, and epoch counters and conditionally updates max epochs, last learning rate and its seen flag, last epoch duration, first and best loss, and per-session step counts. When the update represents a training step, records the step timing and increments steps_this_session.
    
    Parameters:
        update (TrainingUpdate): Structured training update containing fields such as
            step, loss, epoch, max_epochs, lr, epoch_time, and kind.
        stats (TrainingStats): Mutable statistics object that will be updated in place.
    """
    stats.current_step = update.step
    stats.last_loss = update.loss
    stats.current_epoch = update.epoch
    if update.max_epochs > 0:
        stats.max_epochs = update.max_epochs
    if update.lr >= 0 and update.kind == "step":
        stats.last_lr = update.lr
        stats._lr_seen = True
    if update.epoch_time > 0:
        stats.last_epoch_time = update.epoch_time

    if stats.first_loss == 0.0 and update.loss > 0:
        stats.first_loss = update.loss
    if update.loss > 0 and update.loss < stats.best_loss:
        stats.best_loss = update.loss

    if update.kind == "step":
        stats.record_step()
        stats.steps_this_session += 1


def _process_tuple(step: int, loss: float, msg: str, stats: TrainingStats) -> None:
    """
    Update a TrainingStats object with values extracted from a raw (step, loss, msg) tuple by parsing the message.
    
    Parses epoch and epoch-time information from common training log message patterns and updates the following fields on `stats` in place: `current_step`, `last_loss`, `first_loss` (if not yet set), `best_loss`, `current_epoch`, `max_epochs`, `last_epoch_time`, and step-tracking via `record_step()` which increments `steps_this_session`.
    
    Parameters:
        step (int): Current global step index from the training loop.
        loss (float): Reported loss value for this step.
        msg (str): Human-readable training message; may contain epoch info like "Epoch X/Y" and timing like "in Z.Zs".
        stats (TrainingStats): Mutable training statistics accumulator to be updated.
    """
    stats.current_step = step
    stats.last_loss = loss

    if stats.first_loss == 0.0 and loss > 0:
        stats.first_loss = loss
    if loss > 0 and loss < stats.best_loss:
        stats.best_loss = loss

    # Parse epoch from message patterns:
    #   "Epoch 15/100, Step 450, Loss: 0.7234"
    #   "[OK] Epoch 15/100 in 23.4s, Loss: 0.7234"
    msg_lower = msg.lower()
    if "epoch" in msg_lower:
        try:
            # Find "Epoch X/Y" pattern
            idx = msg.lower().index("epoch")
            rest = msg[idx + 5:].strip()
            parts = rest.split("/")
            if len(parts) >= 2:
                epoch_num = int(parts[0].strip())
                max_part = parts[1].split(",")[0].split(" ")[0].strip()
                max_epochs = int(max_part)
                stats.current_epoch = epoch_num
                if max_epochs > 0:
                    stats.max_epochs = max_epochs
        except (ValueError, IndexError):
            pass

    # Parse epoch time from "[OK] Epoch X/Y in Z.Zs"
    if " in " in msg and ("s," in msg or msg.rstrip().endswith("s")):
        try:
            time_part = msg.split(" in ")[1].split("s")[0].strip()
            stats.last_epoch_time = float(time_part)
        except (IndexError, ValueError):
            pass

    # Detect step messages vs epoch messages for speed tracking
    if msg.startswith("Epoch") and "Step" in msg and "Loss" in msg:
        stats.record_step()
        stats.steps_this_session += 1