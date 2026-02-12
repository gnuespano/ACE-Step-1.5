"""
Startup banner for Side-Step CLI.

Shows a branded header with a random motto (Minecraft-style splash),
subcommand, framework versions, and GPU info.
"""

from __future__ import annotations

import random
import sys
import textwrap
from typing import Optional

from acestep.training_v2.ui import console, is_rich_active

# ---- ASCII logo -------------------------------------------------------------

_LOGO = textwrap.dedent(r"""
  ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████
  ██      ██ ██   ██ ██            ██         ██    ██      ██   ██
  ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████
       ██ ██ ██   ██ ██                 ██    ██    ██      ██
  ███████ ██ ██████  ███████       ███████    ██    ███████ ██     
    """).strip()

# ---- Splash mottos (randomly picked each launch) ----------------------------

_MOTTOS = [
    "Because Gradio is the spawn of Satan.",
    "Sidestepping the spaghetti code.",
    "Bypassing the BS, one epoch at a time.",
    "Research grade? No. dernet grade.",
    "Nimrod-tested. Blackwell-approved.",
    "The 5-Euro Heist.",
    "Je suis calibré.",
    "Born in the VRAM trenches.",
    "323k tokens later, we have a snare.",
    "Designed by a Producer. Debugged by a Nimrod.",
    "Because Gradio is the spawn of Satan.",
    "Talk to an LLM for 10h? Hell yeah.",
    "Side-Step: The 17-Hour Speedrun.",
    "Red errors are just decoration.",
    "Importing sanity... ModuleNotFoundError.",
    "One variable, two names, zero documentation.",
    "Because Gradio is the spawn of Satan.",
    "Refactoring the Prometheus experience.",
    "Surgical training for blunt-force code.",
    "2,500 lines of 'Why?'",
    "The browser tab is not the boss of me.",
    "1.5 Terabytes of RAM and still no documentation.",
    "The only way to debug a 2000-line error is to write a 2000-line error.",
]


def _pick_motto() -> str:
    """
    Selects a random startup motto from the available mottos.
    
    Returns:
        str: A randomly chosen motto from the module's motto list.
    """
    return random.choice(_MOTTOS)


def _get_versions() -> dict:
    """
    Collect available version information for Python, PyTorch, and CUDA.
    
    Attempts to detect installed PyTorch and, if available, its CUDA version; records "not installed" for PyTorch when the import fails. Always attempts to record the running Python version.
    
    Returns:
        info (dict): Mapping of component names to version strings. Possible keys include "Python", "PyTorch", and "CUDA".
    """
    info: dict = {}
    try:
        import torch
        info["PyTorch"] = torch.__version__
        if torch.cuda.is_available():
            info["CUDA"] = torch.version.cuda or "n/a"
    except ImportError:
        info["PyTorch"] = "not installed"
    try:
        info["Python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    except Exception:
        pass
    return info


def _get_gpu_line(device: str = "", precision: str = "") -> str:
    """
    Produce a concise GPU description with the device name and optional VRAM size.
    
    Attempts best-effort GPU detection and returns the device name. If total VRAM is available the name is followed by the VRAM in GiB rounded to one decimal (e.g., "GeForce RTX 3080 (10.0 GiB)"). If detection fails, returns "unknown".
    
    Parameters:
        device (str): Requested device identifier (e.g., "cuda:0"); empty string opts into automatic selection.
        precision (str): Requested numeric precision (e.g., "float16"); empty string opts into automatic selection.
    
    Returns:
        str: Detected GPU name with optional VRAM in GiB, or "unknown" if detection was not possible.
    """
    try:
        from acestep.training_v2.gpu_utils import detect_gpu
        gpu = detect_gpu(requested_device=device or "auto", requested_precision=precision or "auto")
        vram_part = ""
        if gpu.vram_total_mb is not None:
            vram_gb = gpu.vram_total_mb / 1024
            vram_part = f"  ({vram_gb:.1f} GiB)"
        return f"{gpu.name}{vram_part}"
    except Exception:
        return "unknown"


# ---- Public API -------------------------------------------------------------

def show_banner(
    subcommand: str,
    device: str = "",
    precision: str = "",
    extra_lines: Optional[list] = None,
) -> None:
    """
    Display the startup banner for the Side-Step CLI.
    
    Constructs a banner containing the ASCII logo, a randomly chosen motto, a human-readable
    description of the provided subcommand, detected Python / PyTorch / CUDA versions,
    an optional precision label, and GPU information. Renders using Rich when available;
    otherwise prints a plain-text banner to stderr.
    
    Parameters:
        subcommand (str): Key or name of the subcommand to display as the CLI mode.
        device (str): Requested device string forwarded to GPU detection (optional).
        precision (str): Precision label to include in the stack line (e.g., "fp16") (optional).
        extra_lines (Optional[list]): Additional lines to append to the banner; each entry
            is printed on its own indented line (optional).
    """
    versions = _get_versions()
    gpu_line = _get_gpu_line(device, precision)
    motto = _pick_motto()

    ver_parts = []
    if "Python" in versions:
        ver_parts.append(f"Python {versions['Python']}")
    if "PyTorch" in versions:
        ver_parts.append(f"PyTorch {versions['PyTorch']}")
    if "CUDA" in versions:
        ver_parts.append(f"CUDA {versions['CUDA']}")
    if precision:
        ver_parts.append(precision)
    ver_str = " | ".join(ver_parts)

    _SUBCOMMAND_DESC = {
        "vanilla": "vanilla (original behaviour, bugged timesteps)",
        "fixed": "fixed (corrected timesteps + CFG dropout)",
        "selective": "selective (dataset-specific module selection)",
        "estimate": "estimate (gradient sensitivity analysis)",
    }
    sub_desc = _SUBCOMMAND_DESC.get(subcommand, subcommand)

    if is_rich_active() and console is not None:
        from rich.panel import Panel
        from rich.text import Text

        body = Text()
        body.append(_LOGO.strip() + "\n", style="bold cyan")
        body.append(f'  "{motto}"\n\n', style="italic yellow")
        body.append("  LoRA Fine-Tuning CLI\n\n", style="dim")
        body.append("  Mode   : ", style="dim")
        body.append(f"{sub_desc}\n", style="bold")
        body.append("  Stack  : ", style="dim")
        body.append(f"{ver_str}\n", style="")
        body.append("  GPU    : ", style="dim")
        body.append(f"{gpu_line}\n", style="")

        if extra_lines:
            for line in extra_lines:
                body.append(f"  {line}\n", style="dim")

        console.print(Panel(body, border_style="cyan", padding=(0, 1)))
    else:
        # Plain text fallback
        print(_LOGO.strip(), file=sys.stderr)
        print(f'  "{motto}"', file=sys.stderr)
        print(f"  Side-Step -- LoRA Fine-Tuning CLI", file=sys.stderr)
        print(f"  Mode   : {sub_desc}", file=sys.stderr)
        print(f"  Stack  : {ver_str}", file=sys.stderr)
        print(f"  GPU    : {gpu_line}", file=sys.stderr)
        if extra_lines:
            for line in extra_lines:
                print(f"  {line}", file=sys.stderr)
        print(file=sys.stderr)