"""
Lean Per-Phase Model Loading for ACE-Step Training V2

Two entry points:
    load_preprocessing_models()  -- VAE + text encoder + condition encoder
    load_decoder_for_training()  -- Full model with decoder accessible

Each function loads only what is needed for its phase, supports torch.no_grad()
context, and provides proper cleanup helpers.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _is_flash_attention_available(device: str) -> bool:
    """
    Determine whether the flash_attn package is importable and the target device is a CUDA device.
    
    Parameters:
        device (str): Device identifier (e.g., "cuda", "cuda:0"). Only devices whose string starts with "cuda" are considered CUDA devices.
    
    Returns:
        `true` if the device string starts with "cuda" and `flash_attn` can be imported, `false` otherwise.
    """
    if not device.startswith("cuda"):
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


# Variant -> subdirectory mapping
_VARIANT_DIR = {
    "turbo": "acestep-v15-turbo",
    "base": "acestep-v15-base",
    "sft": "acestep-v15-sft",
}


def _resolve_model_dir(checkpoint_dir: str | Path, variant: str) -> Path:
    """
    Resolve the model subdirectory for a given variant inside a checkpoint directory.
    
    Parameters:
        checkpoint_dir (str | Path): Root checkpoint directory containing model variant subdirectories.
        variant (str): Variant key identifying the subdirectory (for example "turbo", "base", "sft").
    
    Returns:
        Path: Path to the resolved variant model directory.
    
    Raises:
        ValueError: If the variant is unknown.
        FileNotFoundError: If the resolved model directory does not exist.
    """
    subdir = _VARIANT_DIR.get(variant)
    if subdir is None:
        raise ValueError(f"Unknown model variant: {variant!r}")
    p = Path(checkpoint_dir) / subdir
    if not p.is_dir():
        raise FileNotFoundError(f"Model directory not found: {p}")
    return p


def _resolve_dtype(precision: str) -> torch.dtype:
    """
    Resolve a precision name to the corresponding PyTorch dtype.
    
    Parameters:
        precision (str): Precision identifier, e.g. "bf16", "fp16", or "fp32".
    
    Returns:
        torch.dtype: The matching PyTorch dtype; returns `torch.bfloat16` if the precision is unknown.
    """
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping.get(precision, torch.bfloat16)


def read_model_config(checkpoint_dir: str | Path, variant: str) -> Dict[str, Any]:
    """
    Load the model's config.json for the given checkpoint variant and return it as a dict.
    
    Parameters:
        checkpoint_dir (str | Path): Path to the checkpoint root that contains model variant subdirectories.
        variant (str): Variant key identifying the model subdirectory (e.g., "turbo", "base", "sft").
    
    Returns:
        Dict[str, Any]: Parsed JSON content of the model's config.json.
    
    Raises:
        FileNotFoundError: If the resolved model directory or its config.json file does not exist.
        ValueError: If the provided variant is not recognized.
    """
    model_dir = _resolve_model_dir(checkpoint_dir, variant)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    return json.loads(config_path.read_text())


# ---------------------------------------------------------------------------
# Decoder loading (for training / estimation)
# ---------------------------------------------------------------------------

def load_decoder_for_training(
    checkpoint_dir: str | Path,
    variant: str = "turbo",
    device: str = "cpu",
    precision: str = "bf16",
) -> Any:
    """
    Load the full AceStepConditionGenerationModel prepared for training.
    
    The returned model is moved to the requested device and dtype, set to evaluation mode, and has all parameters frozen (the trainer is expected to enable gradients for LoRA parameters).
    
    Parameters:
        checkpoint_dir (str | Path): Root checkpoints directory containing the model variant.
        variant (str): Variant subdirectory to load; one of "turbo", "base", or "sft".
        device (str): Target device identifier (e.g., "cpu", "cuda:0").
        precision (str): Precision hint; supported values are "bf16", "fp16", and "fp32".
    
    Returns:
        Any: The loaded AceStepConditionGenerationModel instance.
    
    Raises:
        RuntimeError: If the model cannot be loaded with any supported attention implementation.
    """
    from transformers import AutoModel

    model_dir = _resolve_model_dir(checkpoint_dir, variant)
    dtype = _resolve_dtype(precision)

    logger.info("[INFO] Loading model from %s (variant=%s, dtype=%s)", model_dir, variant, dtype)

    # Try attention implementations in preference order.
    # flash_attention_2 first (matches handler.initialize_service), then sdpa, then eager.
    attn_candidates = []
    if _is_flash_attention_available(device):
        attn_candidates.append("flash_attention_2")
    attn_candidates.extend(["sdpa", "eager"])

    model = None
    last_err: Optional[Exception] = None

    for attn_impl in attn_candidates:
        try:
            model = AutoModel.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                attn_implementation=attn_impl,
                dtype=dtype,
            )
            print(f"[OK] Model loaded with attn_implementation={attn_impl}")
            break
        except Exception as exc:
            last_err = exc
            logger.warning("[WARN] Failed with attn_implementation=%s: %s", attn_impl, exc)

    if model is None:
        raise RuntimeError(
            f"Failed to load model from {model_dir}: {last_err}"
        ) from last_err

    # Freeze everything by default -- trainer will unfreeze LoRA params
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device).to(dtype)
    model.eval()

    logger.info("[OK] Model on %s (%s), all params frozen", device, dtype)
    return model


# ---------------------------------------------------------------------------
# Preprocessing models (VAE + text encoder + condition encoder)
# ---------------------------------------------------------------------------

def load_preprocessing_models(
    checkpoint_dir: str | Path,
    variant: str = "turbo",
    device: str = "cpu",
    precision: str = "bf16",
) -> Dict[str, Any]:
    """
    Load models required for preprocessing: the full condition-generation model, an optional VAE, and an optional text encoder with tokenizer.
    
    Parameters:
        checkpoint_dir (str | Path): Path to the checkpoint directory containing model subfolders.
        variant (str): Variant key selecting the model subdirectory (e.g., "turbo", "base", "sft").
        device (str): Target device for model placement (e.g., "cpu", "cuda").
        precision (str): Numeric precision hint ("bf16", "fp16", "fp32").
    
    Returns:
        dict: A mapping with keys:
            - "model": the full AceStepConditionGenerationModel (always present).
            - "vae": AutoencoderOobleck instance if ckpt/vae exists, otherwise None.
            - "text_tokenizer": HuggingFace tokenizer if ckpt/Qwen3-Embedding-0.6B exists, otherwise None.
            - "text_encoder": Qwen3 text encoder model if ckpt/Qwen3-Embedding-0.6B exists, otherwise None.
    
    Notes:
        Call cleanup_preprocessing_models(...) to free resources when the returned models are no longer needed.
    """
    from transformers import AutoModel, AutoTokenizer
    from diffusers.models import AutoencoderOobleck

    ckpt = Path(checkpoint_dir)
    dtype = _resolve_dtype(precision)
    result: Dict[str, Any] = {}

    # 1. Full model (needed for condition encoder)
    model = load_decoder_for_training(checkpoint_dir, variant, device, precision)
    result["model"] = model

    # 2. VAE
    vae_path = ckpt / "vae"
    if vae_path.is_dir():
        vae = AutoencoderOobleck.from_pretrained(str(vae_path))
        vae = vae.to(device).to(dtype)
        vae.eval()
        result["vae"] = vae
        logger.info("[OK] VAE loaded from %s", vae_path)
    else:
        result["vae"] = None
        logger.warning("[WARN] VAE directory not found: %s", vae_path)

    # 3. Text encoder + tokenizer
    text_path = ckpt / "Qwen3-Embedding-0.6B"
    if text_path.is_dir():
        result["text_tokenizer"] = AutoTokenizer.from_pretrained(str(text_path))
        text_enc = AutoModel.from_pretrained(str(text_path))
        text_enc = text_enc.to(device).to(dtype)
        text_enc.eval()
        result["text_encoder"] = text_enc
        logger.info("[OK] Text encoder loaded from %s", text_path)
    else:
        result["text_tokenizer"] = None
        result["text_encoder"] = None
        logger.warning("[WARN] Text encoder directory not found: %s", text_path)

    return result


def cleanup_preprocessing_models(models: Dict[str, Any]) -> None:
    """
    Free memory used by preprocessing models and clear related caches.
    
    This function mutates `models` in place by removing stored model objects, attempting to move objects that implement `.to()` to CPU, deleting references, running garbage collection, and clearing the CUDA cache if available. Use this to explicitly release memory after preprocessing steps.
    
    Parameters:
        models (Dict[str, Any]): Mapping of model names to model objects; all entries will be removed from this dict.
    """
    for key in list(models.keys()):
        obj = models.pop(key, None)
        if obj is not None and hasattr(obj, "to"):
            try:
                obj.to("cpu")
            except Exception:
                pass
        del obj

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[OK] Preprocessing models cleaned up")