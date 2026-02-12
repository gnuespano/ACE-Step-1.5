"""
Reusable Rich/fallback prompt helpers for the interactive wizard.

Provides menu selection, typed value prompts, path prompts, boolean prompts,
and section headers -- with automatic Rich fallback to plain ``input()``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from acestep.training_v2.ui import console, is_rich_active

# Windows uses spawn-based multiprocessing which breaks DataLoader workers
IS_WINDOWS = sys.platform == "win32"
DEFAULT_NUM_WORKERS = 0 if IS_WINDOWS else 4


# ---- Helpers ----------------------------------------------------------------

def menu(
    title: str,
    options: list[tuple[str, str]],
    default: int = 1,
) -> str:
    """Display a numbered menu and return the chosen key.

    Args:
        title: Prompt text.
        options: List of ``(key, label)`` tuples.
        default: 1-based default index.

    Returns:
        The ``key`` of the chosen option.
    """
    if is_rich_active() and console is not None:
        console.print()
        console.print(f"  [bold]{title}[/]\n")
        for i, (key, label) in enumerate(options, 1):
            marker = "[bold cyan]>[/]" if i == default else " "
            tag = "  [dim](default)[/]" if i == default else ""
            console.print(f"    {marker} [bold]{i}[/]. {label}{tag}")
        console.print()

        from rich.prompt import IntPrompt
        while True:
            choice = IntPrompt.ask(
                "  Choice",
                default=default,
                console=console,
            )
            if 1 <= choice <= len(options):
                return options[choice - 1][0]
            console.print(f"  [red]Please enter a number between 1 and {len(options)}[/]")
    else:
        print(f"\n  {title}\n")
        for i, (key, label) in enumerate(options, 1):
            tag = " (default)" if i == default else ""
            print(f"    {i}. {label}{tag}")
        print()
        while True:
            try:
                raw = input(f"  Choice [{default}]: ").strip()
                choice = int(raw) if raw else default
                if 1 <= choice <= len(options):
                    return options[choice - 1][0]
                print(f"  Please enter a number between 1 and {len(options)}")
            except ValueError:
                print(f"  Please enter a number between 1 and {len(options)}")


def ask(
    label: str,
    default: Any = None,
    required: bool = False,
    type_fn: type = str,
    choices: Optional[list] = None,
) -> Any:
    """
    Prompt the user for a single value, applying an optional default, type casting, and choice validation.
    
    Parameters:
        label (str): Prompt text displayed to the user.
        default (Any, optional): Value returned when the user accepts the default or presses Enter for optional prompts. If None and `required` is True, empty input is rejected.
        required (bool): If True, empty input is not allowed and the prompt will repeat until a non-empty value is provided.
        type_fn (type or callable): Function used to cast the input to a desired type (for example `str`, `int`, or `float`).
        choices (list, optional): If provided, the input must match one of these values (comparison performed using their string representations).
    
    Returns:
        Any: The user's input, cast to `type_fn`. Returns `None` only when the prompt is optional and the user submits an empty response.
    """
    if choices:
        choice_str = f" ({'/'.join(str(c) for c in choices)})"
    else:
        choice_str = ""

    if is_rich_active() and console is not None:
        from rich.prompt import Prompt, IntPrompt, FloatPrompt

        prompt_cls = Prompt
        if type_fn is int:
            prompt_cls = IntPrompt
        elif type_fn is float:
            prompt_cls = FloatPrompt

        while True:
            result = prompt_cls.ask(
                f"  {label}{choice_str}",
                default=default if default is not None else ...,
                console=console,
            )
            if result is ...:
                if required:
                    console.print("  [red]This field is required[/]")
                    continue
                return None  # optional field, user pressed Enter to skip
            if required and not str(result).strip():
                console.print("  [red]This field is required[/]")
                continue
            if choices and str(result) not in [str(c) for c in choices]:
                console.print(f"  [red]Must be one of: {', '.join(str(c) for c in choices)}[/]")
                continue
            return type_fn(result) if not isinstance(result, type_fn) else result
    else:
        default_str = f" [{default}]" if default is not None else ""
        while True:
            raw = input(f"  {label}{choice_str}{default_str}: ").strip()
            if not raw and default is not None:
                return default
            if not raw and required:
                print("  This field is required")
                continue
            try:
                val = type_fn(raw)
                if choices and str(val) not in [str(c) for c in choices]:
                    print(f"  Must be one of: {', '.join(str(c) for c in choices)}")
                    continue
                return val
            except (ValueError, TypeError):
                print(f"  Invalid input, expected {type_fn.__name__}")


def ask_path(
    label: str,
    default: Optional[str] = None,
    must_exist: bool = False,
) -> str:
    """Ask for a filesystem path, optionally validating existence."""
    while True:
        val = ask(label, default=default, required=True)
        if must_exist and not Path(val).exists():
            if is_rich_active() and console is not None:
                console.print(f"  [red]Path not found: {val}[/]")
            else:
                print(f"  Path not found: {val}")
            continue
        return val


def ask_bool(label: str, default: bool = True) -> bool:
    """
    Prompt the user for a yes/no response.
    
    Parameters:
        label (str): The prompt text shown to the user.
        default (bool): The default choice used when the user presses Enter (`True` means "yes", `False` means "no").
    
    Returns:
        True if the user's response is a yes-like value (`"yes"`, `"y"`, `"true"`, `"1"`), False otherwise.
    """
    choices = ["yes", "no"]
    default_str = "yes" if default else "no"
    result = ask(label, default=default_str, choices=choices)
    return result.lower() in ("yes", "y", "true", "1")


def section(title: str) -> None:
    """
    Prints a formatted section header; uses rich styling when available.
    """
    if is_rich_active() and console is not None:
        console.print(f"\n  [bold cyan]--- {title} ---[/]\n")
    else:
        print(f"\n  --- {title} ---\n")