"""
Error presentation for the ACE-Step Training V2 CLI.

Wraps exceptions in Rich Panels with actionable suggestions.
Falls back to plain ``[FAIL]`` messages when Rich is unavailable.
"""

from __future__ import annotations

import sys
import traceback
from typing import Optional

from acestep.training_v2.ui import console, is_rich_active

# ---- Suggestion database ----------------------------------------------------

_SUGGESTIONS: dict[str, list[str]] = {
    # CUDA OOM
    "CUDA out of memory": [
        "Reduce --batch-size (try 1)",
        "Reduce --rank (try 32 or 16)",
        "Reduce --gradient-accumulation",
        "Close other GPU-consuming processes",
    ],
    "OutOfMemoryError": [
        "Reduce --batch-size (try 1)",
        "Reduce --rank (try 32 or 16)",
    ],
    # Model loading
    "No such file or directory": [
        "Check --checkpoint-dir points to the correct path",
        "Verify the model variant directory exists (e.g. acestep-v15-turbo/)",
    ],
    "not found": [
        "Check that all required paths exist",
        "Run preprocessing first if .pt files are missing",
    ],
    # LoRA / PEFT
    "peft": [
        "Install PEFT:  pip install peft",
    ],
    "PeftModel": [
        "Install PEFT:  pip install peft",
    ],
    # Fabric
    "lightning": [
        "Install Lightning Fabric:  pip install lightning",
        "Or the trainer will fall back to a basic PyTorch loop",
    ],
    # Flash attention
    "flash_attn": [
        "Flash Attention 2 is optional. The model will fall back to SDPA",
        "To install:  pip install flash-attn --no-build-isolation",
    ],
    # Dtype
    "bfloat16": [
        "Your GPU may not support bf16.  Try --precision fp16",
    ],
    # Generic
    "Permission denied": [
        "Check file/directory permissions on --output-dir",
    ],
    "No space left on device": [
        "Free up disk space or change --output-dir to a different drive",
    ],
}


def _find_suggestions(error_str: str) -> list[str]:
    """
    Return suggestion tips whose patterns appear in the provided error text.
    
    Performs case-insensitive substring matching of `error_str` against the keys in the module-level `_SUGGESTIONS` mapping and aggregates all suggestion tips for every matching pattern.
    
    Parameters:
        error_str (str): The error message or text to scan for known pattern keys (matching is case-insensitive and uses substring containment).
    
    Returns:
        list[str]: A list of suggestion strings for all patterns found in `error_str`. May be empty if no patterns match.
    """
    suggestions: list[str] = []
    lower = error_str.lower()
    for pattern, tips in _SUGGESTIONS.items():
        if pattern.lower() in lower:
            suggestions.extend(tips)
    return suggestions


# ---- Public API -------------------------------------------------------------

def handle_error(
    exc: BaseException,
    context: str = "Training",
    show_traceback: bool = False,
) -> None:
    """
    Display an error message labeled by `context`, include any suggested fixes matched from the error text, and optionally show the traceback.
    
    Parameters:
        exc (BaseException): The exception to present.
        context (str): Short label for where the error occurred (e.g., "Model loading", "Training").
        show_traceback (bool): If True, include the formatted traceback in the output.
    """
    error_str = str(exc)
    exc_type = type(exc).__name__
    suggestions = _find_suggestions(error_str)

    if is_rich_active() and console is not None:
        _show_rich(exc, exc_type, error_str, context, suggestions, show_traceback)
    else:
        _show_plain(exc, exc_type, error_str, context, suggestions, show_traceback)


def show_warning(msg: str) -> None:
    """
    Show a warning message using Rich styling when available, otherwise print the message to stderr prefixed with `[WARN]`.
    
    Parameters:
        msg (str): Warning text to display.
    """
    if is_rich_active() and console is not None:
        console.print(f"[bold yellow][WARN][/] {msg}")
    else:
        print(f"[WARN] {msg}", file=sys.stderr)


def show_fail(msg: str) -> None:
    """
    Display a failure message labeled "[Side-Step]"; uses Rich styling when available, otherwise writes plain text to stderr.
    
    Parameters:
        msg (str): Failure message to display.
    """
    if is_rich_active() and console is not None:
        console.print(f"[bold red][Side-Step][/] {msg}")
    else:
        print(f"[Side-Step] {msg}", file=sys.stderr)


def show_info(msg: str) -> None:
    """Display an info message."""
    if is_rich_active() and console is not None:
        console.print(f"[bold blue][INFO][/] {msg}")
    else:
        print(f"[INFO] {msg}", file=sys.stderr)


def show_error(
    title: str,
    message: str,
    suggestion: Optional[str] = None,
) -> None:
    """
    Show an error panel with a title, message, and optional suggested fix; falls back to plain-text output when Rich is unavailable.
    
    Parameters:
        title (str): Panel title (for example, "Vanilla + Non-Turbo Model Warning").
        message (str): Main error message text; may be multiline.
        suggestion (Optional[str]): Optional suggested fix or command to run.
    """
    if is_rich_active() and console is not None:
        from rich.panel import Panel
        from rich.text import Text

        body = Text()
        body.append(message, style="red")
        if suggestion:
            body.append("\n\nSuggested fix:\n", style="bold yellow")
            body.append(f"  {suggestion}", style="yellow")

        console.print(
            Panel(
                body,
                title=f"[bold red]{title}[/]",
                border_style="red",
                padding=(0, 1),
            )
        )
    else:
        print(f"\n[Side-Step] {title}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print(message, file=sys.stderr)
        if suggestion:
            print(f"\nSuggested fix: {suggestion}", file=sys.stderr)
        print("-" * 60 + "\n", file=sys.stderr)


# ---- Rich rendering --------------------------------------------------------

def _show_rich(
    exc: BaseException,
    exc_type: str,
    error_str: str,
    context: str,
    suggestions: list[str],
    show_traceback: bool,
) -> None:
    """
    Render an exception inside a styled Rich Panel including optional suggested fixes and an optional traceback.
    
    Parameters:
        exc: The caught exception instance to display.
        exc_type: The exception type name to show as a prefix (e.g., "RuntimeError").
        error_str: The exception message or summary to display.
        context: A short label describing the operation or phase that produced the error (used as the panel title).
        suggestions (list[str]): Ordered suggestion strings to display under "Suggested fixes:"; no suggestions are shown if empty.
        show_traceback (bool): If True, appends the formatted traceback to the panel.
        
    Notes:
        This function requires a global Rich console to be available (the function asserts `console is not None`). It uses Rich Text and Panel for styled output.
    """
    from rich.panel import Panel
    from rich.text import Text

    assert console is not None

    body = Text()
    body.append(f"{exc_type}: ", style="bold red")
    body.append(f"{error_str}\n", style="red")

    if suggestions:
        body.append("\nSuggested fixes:\n", style="bold yellow")
        for i, tip in enumerate(suggestions, 1):
            body.append(f"  {i}. {tip}\n", style="yellow")

    if show_traceback:
        body.append("\nTraceback:\n", style="dim")
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        body.append(tb_str, style="dim red")

    console.print(
        Panel(
            body,
            title=f"[bold red]{context} Error[/]",
            border_style="red",
            padding=(0, 1),
        )
    )


# ---- Plain rendering -------------------------------------------------------

def _show_plain(
    exc: BaseException,
    exc_type: str,
    error_str: str,
    context: str,
    suggestions: list[str],
    show_traceback: bool,
) -> None:
    """
    Display a plain-text error message to stderr with optional suggested fixes and traceback.
    
    Parameters:
        exc: The original exception instance.
        exc_type: Human-readable exception type name (e.g., "RuntimeError").
        error_str: Short error message extracted from the exception.
        context: Label describing where the error occurred (e.g., "Training", "Model loading").
        suggestions: A list of suggested fixes to show to the user; enumerated if present.
        show_traceback: If True, prints the full traceback for the provided exception.
    """
    print(f"\n[Side-Step] {context} Error: {exc_type}: {error_str}", file=sys.stderr)
    if suggestions:
        print("       Suggested fixes:", file=sys.stderr)
        for i, tip in enumerate(suggestions, 1):
            print(f"         {i}. {tip}", file=sys.stderr)
    if show_traceback:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    print(file=sys.stderr)