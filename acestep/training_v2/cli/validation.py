"""
Path validation and target-module resolution for ACE-Step Training V2 CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from acestep.training_v2.cli.args import VARIANT_DIR_MAP


def validate_paths(args: argparse.Namespace) -> bool:
    """
    Validate required filesystem paths for the CLI subcommands and emit failure/warning messages to stderr.
    
    Checks:
    - For subcommand "compare-configs": verifies every path in `args.configs` is a file.
    - For other subcommands: verifies `args.checkpoint_dir` exists, `args.model_variant` maps to a known subdirectory under the checkpoint directory, and that the corresponding model directory exists. If present, verifies `args.dataset_dir` is a directory. If `args.resume_from` is provided but missing, emits a warning and continues.
    
    Parameters:
        args (argparse.Namespace): Namespace containing command arguments. Expected attributes:
            - subcommand (str)
            - configs (Iterable[str]) when subcommand == "compare-configs"
            - checkpoint_dir (str)
            - model_variant (str)
            - dataset_dir (Optional[str])
            - resume_from (Optional[str])
    
    Returns:
        bool: `True` if all required checks pass, `False` on the first failing check.
    
    Side effects:
        Writes "[FAIL] ..." messages to stderr for fatal validation failures and "[WARN] ..." for a missing resume path.
    """
    sub = args.subcommand

    if sub == "compare-configs":
        for p in args.configs:
            if not Path(p).is_file():
                print(f"[FAIL] Config file not found: {p}", file=sys.stderr)
                return False
        return True

    # All other subcommands need checkpoint-dir
    ckpt_root = Path(args.checkpoint_dir)
    if not ckpt_root.is_dir():
        print(f"[FAIL] Checkpoint directory not found: {ckpt_root}", file=sys.stderr)
        return False

    variant_dir = VARIANT_DIR_MAP.get(args.model_variant)
    if variant_dir is None:
        print(f"[FAIL] Unknown model variant: {args.model_variant}", file=sys.stderr)
        return False

    model_dir = ckpt_root / variant_dir
    if not model_dir.is_dir():
        print(
            f"[FAIL] Model directory not found: {model_dir}\n"
            f"       Expected subdirectory '{variant_dir}' under {ckpt_root}",
            file=sys.stderr,
        )
        return False

    # Dataset dir
    ds_dir = getattr(args, "dataset_dir", None)
    if ds_dir is not None and not Path(ds_dir).is_dir():
        print(f"[FAIL] Dataset directory not found: {ds_dir}", file=sys.stderr)
        return False

    # Resume path
    resume = getattr(args, "resume_from", None)
    if resume is not None and not Path(resume).exists():
        print(f"[WARN] Resume path not found (will train from scratch): {resume}", file=sys.stderr)

    return True


def resolve_target_modules(target_modules: list, attention_type: str) -> list:
    """
    Resolve module pattern names according to the specified attention type.
    
    When `attention_type` is "self" or "cross", each module name that does not already contain a dot is prefixed with `self_attn.` or `cross_attn.` respectively. Module names that already contain a dot are preserved. If `attention_type` is "both" or is unrecognized, the input list is returned unchanged.
    
    Returns:
        list: Resolved module patterns with attention-type prefixes applied where appropriate.
    """
    if attention_type == "both":
        return target_modules

    prefix_map = {
        "self": "self_attn",
        "cross": "cross_attn",
    }
    prefix = prefix_map.get(attention_type)
    if prefix is None:
        return target_modules

    resolved = []
    for mod in target_modules:
        if "." in mod:
            resolved.append(mod)
        else:
            resolved.append(f"{prefix}.{mod}")

    return resolved