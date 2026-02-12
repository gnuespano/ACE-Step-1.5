"""
Preprocessing -- Reusable Module

Provides ``preprocess_audio_files()`` for use from both the CLI and the TUI.
Wraps the preprocessing pipeline: encode audio through VAE + text encoder,
save resulting tensors as .pt files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def preprocess_audio_files(
    source_dir: str,
    output_dir: str,
    checkpoint_dir: str,
    variant: str = "turbo",
    max_duration: float = 240.0,
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Preprocess audio files in a directory into .pt tensors containing the waveform and optional VAE latents.
    
    Parameters:
        source_dir (str): Directory containing audio files (.mp3, .wav, .flac, .ogg, .m4a).
        output_dir (str): Directory where .pt output files will be written (created if missing).
        checkpoint_dir (str): Path to ACE-Step model checkpoints used for preprocessing.
        variant (str): Model variant to load (e.g., "turbo", "base", "sft").
        max_duration (float): Maximum audio duration in seconds; audio longer than this is truncated.
        progress_callback (Optional[Callable]): Optional function called as progress_callback(current, total, filename).
        cancel_check (Optional[Callable]): Optional zero-argument function that should return True to cancel processing.
    
    Returns:
        dict: Summary with keys:
            - processed (int): Number of files successfully preprocessed or skipped because output existed.
            - failed (int): Number of files that failed to preprocess.
            - total (int): Total number of audio files found and considered.
            - output_dir (str): Path to the output directory where .pt files were written.
    """
    import torch

    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect audio files
    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    audio_files = [
        f for f in sorted(source_path.iterdir())
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        logger.warning("[Side-Step] No audio files found in %s", source_dir)
        return {"processed": 0, "failed": 0, "total": 0, "output_dir": str(output_path)}

    total = len(audio_files)
    logger.info("[Side-Step] Found %d audio files to preprocess", total)

    # Load preprocessing models
    from acestep.training_v2.model_loader import (
        load_preprocessing_models,
        cleanup_preprocessing_models,
    )
    from acestep.training_v2.gpu_utils import detect_gpu

    gpu = detect_gpu()
    models = load_preprocessing_models(
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=gpu.device,
        precision=gpu.precision,
    )

    processed = 0
    failed = 0

    try:
        for i, audio_file in enumerate(audio_files):
            if cancel_check and cancel_check():
                logger.info("[Side-Step] Preprocessing cancelled at %d/%d", i, total)
                break

            if progress_callback:
                progress_callback(i, total, audio_file.name)

            try:
                output_file = output_path / f"{audio_file.stem}.pt"
                if output_file.exists():
                    logger.info("[Side-Step] Skipping (exists): %s", audio_file.name)
                    processed += 1
                    continue

                # Load audio
                import torchaudio
                waveform, sr = torchaudio.load(str(audio_file))

                # Truncate to max duration
                max_samples = int(max_duration * sr)
                if waveform.shape[-1] > max_samples:
                    waveform = waveform[..., :max_samples]

                # Resample to model's expected rate if needed (24kHz)
                if sr != 24000:
                    resampler = torchaudio.transforms.Resample(sr, 24000)
                    waveform = resampler(waveform)

                # Mono downmix if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Encode through VAE (if models available)
                device = gpu.device
                dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
                dtype = dtype_map.get(gpu.precision, torch.bfloat16)

                with torch.no_grad():
                    waveform = waveform.to(device, dtype=dtype)

                    tensor_data = {"waveform": waveform.cpu()}

                    # Encode through available models
                    if "vae" in models and models["vae"] is not None:
                        vae = models["vae"]
                        latents = vae.encode(waveform.unsqueeze(0))
                        if hasattr(latents, "latent_dist"):
                            latents = latents.latent_dist.sample()
                        tensor_data["target_latents"] = latents.squeeze(0).cpu()

                torch.save(tensor_data, output_file)
                processed += 1
                logger.info("[Side-Step] Preprocessed: %s", audio_file.name)

            except Exception as e:
                failed += 1
                logger.error("[Side-Step] Failed to preprocess %s: %s", audio_file.name, e)

        if progress_callback:
            progress_callback(total, total, "Done")

    finally:
        cleanup_preprocessing_models(models)

    result = {
        "processed": processed,
        "failed": failed,
        "total": total,
        "output_dir": str(output_path),
    }
    logger.info("[Side-Step] Preprocessing complete: %d/%d processed, %d failed",
                processed, total, failed)
    return result