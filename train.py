#!/usr/bin/env python3
"""
ACE-Step Training V2 -- CLI Entry Point

Usage:
    python train.py <subcommand> [args]

Subcommands:
    vanilla          Reproduce existing (bugged) training for backward compatibility
    fixed            Corrected training: continuous timesteps + CFG dropout
    selective        Corrected training with dataset-specific module selection
    estimate         Gradient sensitivity analysis (no training)
    compare-configs  Compare module config JSON files

Examples:
    python train.py fixed --checkpoint-dir ./checkpoints --model-variant turbo \\
        --dataset-dir ./preprocessed_tensors/jazz --output-dir ./lora_output/jazz

    python train.py --help
"""

from __future__ import annotations

import logging
import sys

# ---------------------------------------------------------------------------
# Logging setup (before any library imports that might configure logging)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def _has_subcommand() -> bool:
    """
    Determine whether command-line arguments include a known subcommand or a help flag.
    
    Returns:
        bool: `True` if a known subcommand (`vanilla`, `fixed`, `selective`, `estimate`, `compare-configs`)
        or a help flag (`--help` or `-h`) is present, `False` otherwise.
    """
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        return True  # let argparse handle help
    known = {"vanilla", "fixed", "selective", "estimate", "compare-configs"}
    return bool(known & set(args))


def main() -> int:
    # -- Interactive wizard when no subcommand is given -----------------------
    """
    Entry point for the ACE-Step Training V2 CLI that runs an interactive wizard when no subcommand is provided or parses and dispatches CLI subcommands.
    
    When no subcommand is present, launches the interactive wizard and uses its result as the dispatch arguments. When a subcommand is present, builds the root argument parser and parses CLI arguments. Handles a not-yet-implemented preprocess flow (prints guidance and exits). Dispatches to subcommand handlers: `vanilla`, `fixed`, `selective`, `estimate`, and `compare-configs` (the latter performs its own validation). Unknown subcommands and path-validation failures produce a nonzero exit code.
    
    Returns:
        int: Process exit code — `0` for successful or intentionally handled exits (including preprocess and unimplemented placeholders), `1` for validation failures or unknown subcommands, or the integer exit code returned by a delegated subcommand handler.
    """
    if not _has_subcommand():
        from acestep.training_v2.ui.wizard import run_wizard

        args = run_wizard()
        if args is None:
            return 0
    else:
        from acestep.training_v2.cli.common import build_root_parser
        parser = build_root_parser()
        args = parser.parse_args()

    from acestep.training_v2.cli.common import validate_paths

    # -- Preprocessing (wizard only, not yet implemented) --------------------
    if getattr(args, "preprocess", False):
        print("[INFO] Preprocessing is not yet implemented.")
        print("[INFO] Use the Gradio UI or manual scripts to preprocess audio,")
        print("[INFO] then run:  python train.py fixed --dataset-dir <tensor_dir> ...")
        return 0

    # -- Dispatch -----------------------------------------------------------
    sub = args.subcommand

    # compare-configs has its own validation
    if sub == "compare-configs":
        return _run_compare_configs(args)

    # All other subcommands need path validation
    if not validate_paths(args):
        return 1

    if sub == "vanilla":
        from acestep.training_v2.cli.train_vanilla import run_vanilla
        return run_vanilla(args)

    elif sub == "fixed":
        from acestep.training_v2.cli.train_fixed import run_fixed
        return run_fixed(args)

    elif sub == "selective":
        return _run_selective(args)

    elif sub == "estimate":
        return _run_estimate(args)

    else:
        print(f"[FAIL] Unknown subcommand: {sub}", file=sys.stderr)
        return 1


# ===========================================================================
# Placeholder subcommands (Conversation C / D)
# ===========================================================================

def _run_selective(args) -> int:
    """
    Inform the user that selective training is not implemented and suggest alternatives.
    
    Parameters:
        args (argparse.Namespace): Parsed command-line arguments (unused).
    
    Returns:
        int: `0` on successful completion.
    """
    print("[INFO] Selective training is not yet implemented.")
    print("[INFO] Use 'fixed' for corrected training, or 'estimate' for module analysis.")
    return 0


def _run_estimate(args) -> int:
    """
    Inform the user that the gradient estimation subcommand is not implemented.
    
    Parameters:
        args (argparse.Namespace): Parsed command-line arguments (unused).
    
    Returns:
        int: Exit code 0 indicating successful completion of the placeholder.
    """
    print("[INFO] Estimation is not yet implemented.")
    return 0


def _run_compare_configs(args) -> int:
    """
    Validate CLI paths and run the compare-configs subcommand (placeholder).
    
    Parameters:
        args: Parsed command-line arguments for the compare-configs subcommand. Paths referenced in `args` are validated before proceeding.
    
    Returns:
        int: Exit code where `0` indicates the placeholder ran (feature not implemented) and `1` indicates path validation failed.
    """
    from acestep.training_v2.cli.common import validate_paths
    if not validate_paths(args):
        return 1
    print("[INFO] compare-configs is not yet implemented.")
    return 0


# ===========================================================================
# Entry
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())