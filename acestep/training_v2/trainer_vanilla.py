"""
VanillaTrainer -- Thin adapter wrapping the original LoRATrainer for TUI use.

The original ``acestep/training/trainer.py`` ``LoRATrainer`` requires a
``dit_handler`` shim.  This module provides a ``VanillaTrainer`` class with
the same interface as ``FixedTrainer`` so both can be used interchangeably
from the TUI training monitor.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple

import torch

from acestep.training.configs import LoRAConfig, TrainingConfig
from acestep.training.trainer import LoRATrainer
from acestep.training_v2.model_loader import load_decoder_for_training

logger = logging.getLogger(__name__)


class _HandlerShim:
    """Minimal shim satisfying the ``LoRATrainer`` constructor.

    Wraps a decoder model, device string, and dtype so that the upstream
    ``LoRATrainer`` receives the interface it expects from a full
    ``dit_handler``.

    Args:
        model: The decoder ``torch.nn.Module`` to train.
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``).
        dtype: Torch dtype for mixed-precision (e.g. ``torch.bfloat16``).
    """

    def __init__(self, model: torch.nn.Module, device: str, dtype: torch.dtype) -> None:
        """
        Initialize the handler shim that provides the minimal interface expected by LoRATrainer.
        
        Parameters:
            model (torch.nn.Module): The decoder model instance to be used for training.
            device (str): Target device identifier (e.g., "cpu", "cuda").
            dtype (torch.dtype): Desired tensor dtype for training (e.g., torch.bfloat16, torch.float16).
        
        Notes:
            - A `quantization` attribute is created and initialized to `None` as a placeholder for optional quantization metadata.
        """
        self.model = model
        self.device = device
        self.dtype = dtype
        self.quantization = None


class VanillaTrainer:
    """Adapter that wraps the upstream ``LoRATrainer`` to match FixedTrainer's interface.

    Args:
        lora_config: LoRA hyper-parameters (rank, alpha, dropout, targets).
        training_config: Training hyper-parameters (LR, epochs, batch size, ...).
        progress_callback: Optional callable invoked after each upstream
            training update.  Signature:
            ``(epoch: int, step: int, loss: float, lr: float, is_epoch_end: bool) -> Optional[bool]``.
            Return ``False`` to request early stopping.

    Attributes:
        lora_config: Stored LoRA configuration.
        training_config: Stored training configuration.
        progress_callback: Stored callback (may be ``None``).
    """

    def __init__(
        self,
        lora_config: Any,
        training_config: Any,
        progress_callback: Optional[Callable[..., Optional[bool]]] = None,
    ) -> None:
        """
        Create a VanillaTrainer that adapts an upstream LoRATrainer to a FixedTrainer-like interface.
        
        Parameters:
            lora_config: LoRA hyper-parameter structure (e.g., rank, alpha, dropout, target_modules, bias).
            training_config: Training hyper-parameter structure (e.g., learning rate, epochs, batch size, device, precision, dataset paths).
            progress_callback: Optional callable invoked after each upstream training update with signature
                (epoch: int, step: int, loss: float, lr: float, is_epoch_end: bool) -> Optional[bool].
                If it returns `False`, training will stop early.
        """
        self.lora_config = lora_config
        self.training_config = training_config
        self.progress_callback = progress_callback

    # -- Private helpers ---------------------------------------------------

    def _build_configs(self) -> Tuple[LoRAConfig, TrainingConfig, int]:
        """
        Convert V2-style trainer and LoRA settings into base LoRAConfig and TrainingConfig objects.
        
        Returns:
            A tuple (lora_cfg, train_cfg, num_workers):
            - lora_cfg (LoRAConfig): LoRA hyperparameter object populated from self.lora_config with sensible defaults.
            - train_cfg (TrainingConfig): Training hyperparameter object populated from self.training_config with sensible defaults.
            - num_workers (int): Number of DataLoader workers (clamped to 0 on Windows).
        """
        cfg = self.training_config

        lora_cfg = LoRAConfig(
            r=getattr(self.lora_config, "rank", 64),
            alpha=getattr(self.lora_config, "alpha", 128),
            dropout=getattr(self.lora_config, "dropout", 0.0),
            target_modules=getattr(self.lora_config, "target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
            bias=getattr(self.lora_config, "bias", "none"),
        )

        # Windows uses spawn-based multiprocessing which breaks DataLoader workers
        num_workers = getattr(cfg, "num_workers", 4)
        if sys.platform == "win32" and num_workers > 0:
            logger.info("[Side-Step] Windows detected -- setting num_workers=0 (spawn incompatible)")
            num_workers = 0

        train_cfg = TrainingConfig(
            learning_rate=getattr(cfg, "learning_rate", 1e-4),
            batch_size=getattr(cfg, "batch_size", 1),
            gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 4),
            max_epochs=getattr(cfg, "max_epochs", getattr(cfg, "epochs", 100)),
            warmup_steps=getattr(cfg, "warmup_steps", 500),
            weight_decay=getattr(cfg, "weight_decay", 0.01),
            max_grad_norm=getattr(cfg, "max_grad_norm", 1.0),
            seed=getattr(cfg, "seed", 42),
            output_dir=getattr(cfg, "output_dir", "./lora_output"),
            save_every_n_epochs=getattr(cfg, "save_every_n_epochs", 10),
            num_workers=num_workers,
            pin_memory=getattr(cfg, "pin_memory", True),
        )

        return lora_cfg, train_cfg, num_workers

    @staticmethod
    def _setup_device_and_model(
        cfg: Any,
    ) -> Tuple[Any, torch.dtype, torch.nn.Module]:
        """
        Detects the target GPU and loads the decoder model configured for training.
        
        Parameters:
            cfg (object): Configuration object providing:
                - device (str, optional): requested device ("auto" by default).
                - precision (str, optional): requested precision ("auto" by default; e.g., "bf16", "fp16", "fp32").
                - checkpoint_dir (str, optional): path to decoder checkpoints (default "./checkpoints").
                - model_variant or variant (str, optional): model variant name (default "turbo").
        
        Returns:
            tuple: A 3-tuple (gpu, dtype, model) where
                - gpu: detected GPU/context information (includes at least `.device` and `.precision`),
                - dtype (torch.dtype): floating-point dtype selected for training,
                - model (torch.nn.Module): loaded decoder model ready for training.
        """
        from acestep.training_v2.gpu_utils import detect_gpu

        gpu = detect_gpu(
            requested_device=getattr(cfg, "device", "auto"),
            requested_precision=getattr(cfg, "precision", "auto"),
        )
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map.get(gpu.precision, torch.bfloat16)

        model = load_decoder_for_training(
            checkpoint_dir=getattr(cfg, "checkpoint_dir", "./checkpoints"),
            variant=getattr(cfg, "model_variant", getattr(cfg, "variant", "turbo")),
            device=gpu.device,
            precision=gpu.precision,
        )

        return gpu, dtype, model

    @staticmethod
    def _run_training(
        model: torch.nn.Module,
        gpu: Any,
        dtype: torch.dtype,
        lora_cfg: LoRAConfig,
        train_cfg: TrainingConfig,
        cfg: Any,
    ) -> Generator:
        """
        Yield training progress update tuples produced during LoRA training.
        
        Yields:
            Tuples representing training progress updates, typically beginning with
            (step: int, loss: float, ...) and optionally including additional info
            such as messages or metadata.
        """
        handler = _HandlerShim(model=model, device=gpu.device, dtype=dtype)
        trainer = LoRATrainer(handler, lora_cfg, train_cfg)
        dataset_dir = getattr(cfg, "dataset_dir", "")
        resume_from = getattr(cfg, "resume_from", None)

        yield from trainer.train_from_preprocessed(
            tensor_dir=dataset_dir,
            resume_from=resume_from,
        )

    # -- Public API --------------------------------------------------------

    def train(self) -> None:
        """
        Run vanilla training and report progress to the configured callback after each training update.
        
        If a progress_callback was provided to the trainer, it is called for every upstream update with the signature
        (epoch: int, step: int, loss: float, lr: float, is_epoch_end: bool) -> Optional[bool]. The callback is invoked with
        epoch set to 0, lr set to 0.0, and is_epoch_end set to False. If the callback returns False, training stops early.
        """
        cfg = self.training_config

        lora_cfg, train_cfg, _num_workers = self._build_configs()
        gpu, dtype, model = self._setup_device_and_model(cfg)

        for update in self._run_training(model, gpu, dtype, lora_cfg, train_cfg, cfg):
            if self.progress_callback:
                # Upstream yields (step, loss, msg) tuples
                step, loss, _msg = update if len(update) == 3 else (update[0], update[1], "")
                should_continue = self.progress_callback(
                    epoch=0, step=step, loss=loss, lr=0.0, is_epoch_end=False,
                )
                if should_continue is False:
                    break