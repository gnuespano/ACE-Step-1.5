"""
Safe dependency updater for ACE-Step python_embeded environment.

Only installs packages that are completely MISSING from the environment.
Skips high-risk packages (PyTorch, CUDA-specific) that could break the
carefully configured embedded Python setup.

Usage: python update_deps.py
"""

import importlib.metadata
import subprocess
import sys
import os
import re

# These packages must NEVER be auto-installed/updated via pip.
# They require specific CUDA builds or platform-specific wheels
# that are pre-configured in the portable package.
SKIP_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
    "flash-attn",
    "triton",
    "triton-windows",
    "nano-vllm",
    "torchcodec",
    "torchao",
}


def normalize(name):
    """PEP 503 normalize package name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def get_installed():
    """Return set of normalized installed package names."""
    return {normalize(d.name) for d in importlib.metadata.distributions()}


def evaluate_marker(marker_str):
    """
    Evaluate environment marker for current platform.
    Uses packaging.markers if available, otherwise falls back to simple heuristics.
    """
    try:
        from packaging.markers import Marker

        return Marker(marker_str).evaluate()
    except ImportError:
        pass

    # Fallback: simple heuristic for common markers
    marker_lower = marker_str.lower().replace(" ", "")
    current_platform = sys.platform  # 'win32', 'linux', 'darwin'

    # Reject markers explicitly for other platforms
    if current_platform == "win32":
        if "sys_platform=='darwin'" in marker_lower and "!=" not in marker_lower:
            return False
        if "sys_platform=='linux'" in marker_lower and "!=" not in marker_lower:
            return False
        if "sys_platform!='win32'" in marker_lower:
            # Could be "!= 'win32' and != 'darwin'" (means linux)
            return False
        if "platform_machine=='arm64'" in marker_lower:
            return False
    elif current_platform == "darwin":
        if "sys_platform=='win32'" in marker_lower and "!=" not in marker_lower:
            return False
        if "sys_platform=='linux'" in marker_lower and "!=" not in marker_lower:
            return False
    elif current_platform == "linux":
        if "sys_platform=='win32'" in marker_lower and "!=" not in marker_lower:
            return False
        if "sys_platform=='darwin'" in marker_lower and "!=" not in marker_lower:
            return False

    # Default: assume applicable
    return True


def parse_requirements(filepath):
    """
    Parse requirements.txt for packages applicable to current platform.
    Returns list of (raw_name, install_spec) tuples.
    """
    results = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, pip options (--extra-index-url etc.)
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Split off environment marker
            marker_str = None
            if ";" in line:
                spec_part, marker_str = line.rsplit(";", 1)
                spec_part = spec_part.strip()
                marker_str = marker_str.strip()
            else:
                spec_part = line

            # Evaluate marker if present
            if marker_str:
                try:
                    if not evaluate_marker(marker_str):
                        continue  # Not applicable to this platform
                except Exception:
                    continue  # Can't parse marker, skip to be safe

            # Extract package name
            if " @ " in spec_part:
                # URL-based install (e.g., flash-attn @ https://...)
                raw_name = spec_part.split(" @ ")[0].strip()
            else:
                match = re.match(
                    r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)", spec_part
                )
                if match:
                    raw_name = match.group(1)
                else:
                    continue

            # Remove extras bracket for name comparison
            # but keep full spec_part for installation
            base_name = raw_name.split("[")[0]

            results.append((base_name, spec_part))

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(script_dir, "requirements.txt")

    if not os.path.exists(req_file):
        print("[Deps] requirements.txt not found, skipping dependency check.")
        return 0

    print("[Deps] Checking for missing dependencies...")
    print()

    installed = get_installed()
    requirements = parse_requirements(req_file)

    skip_normalized = {normalize(s) for s in SKIP_PACKAGES}

    missing = []
    skipped_missing = []

    for raw_name, install_spec in requirements:
        norm = normalize(raw_name)

        if norm in skip_normalized:
            if norm not in installed:
                skipped_missing.append(raw_name)
            continue

        if norm not in installed:
            missing.append((raw_name, install_spec))

    # Warn about skipped missing packages
    if skipped_missing:
        print("  [!] These core packages are missing but cannot be auto-installed:")
        for name in skipped_missing:
            print(f"      - {name}")
        print(
            "      These require specific CUDA/platform builds."
        )
        print("      Please download the latest portable package to get them.")
        print()

    if not missing:
        print("[Deps] All dependencies are satisfied.")
        return 0

    print(f"[Deps] Found {len(missing)} missing package(s):")
    for name, _ in missing:
        print(f"  + {name}")
    print()

    # Install missing packages one by one
    failed = []
    for name, install_spec in missing:
        print(f"[Deps] Installing {name}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    install_spec,
                    "--disable-pip-version-check",
                    "--no-warn-script-location",
                    "-q",
                    "--no-input",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                print("OK")
            else:
                print("FAILED")
                failed.append(name)
                stderr = result.stderr.strip()
                if stderr:
                    for err_line in stderr.splitlines()[-3:]:
                        print(f"      {err_line}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            failed.append(name)

    print()
    if failed:
        print(
            f"[Deps] Installed {len(missing) - len(failed)}, failed {len(failed)}:"
        )
        for name in failed:
            print(f"  - {name}")
        print("  Please install failed packages manually.")
        return 1
    else:
        print(f"[Deps] All {len(missing)} missing packages installed successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
