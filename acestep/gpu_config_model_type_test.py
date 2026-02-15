"""Unit tests for GPU config model type batch size adjustment."""

import unittest
from unittest.mock import patch, MagicMock

from acestep.gpu_config import (
    GPUConfig,
    update_gpu_config_for_model_type,
    set_global_gpu_config,
    get_global_gpu_config,
    DIT_INFERENCE_VRAM_PER_BATCH,
)


class GpuConfigModelTypeTests(unittest.TestCase):
    """Tests for model-type-aware batch size limit updates."""

    def setUp(self):
        """Reset global GPU config before each test."""
        # Create a test config with 8GB VRAM (typical for user's RTX 4060)
        self.test_config = GPUConfig(
            tier="tier4",
            gpu_memory_gb=8.0,
            max_duration_with_lm=240,
            max_duration_without_lm=360,
            max_batch_size_with_lm=2,
            max_batch_size_without_lm=4,
            init_lm_default=True,
            available_lm_models=["acestep-5Hz-lm-0.6B"],
            recommended_lm_model="acestep-5Hz-lm-0.6B",
            lm_backend_restriction="all",
            recommended_backend="vllm",
            offload_to_cpu_default=True,
            offload_dit_to_cpu_default=True,
            quantization_default=True,
            compile_model_default=True,
            lm_memory_gb={"0.6B": 3},
        )
        set_global_gpu_config(self.test_config)

    def test_base_model_reduces_batch_size_limits(self):
        """Base model (with CFG) should have lower batch limits than turbo."""
        # Initial config assumes turbo model (0.3 GB per batch)
        initial_config = get_global_gpu_config()
        initial_without_lm = initial_config.max_batch_size_without_lm
        initial_with_lm = initial_config.max_batch_size_with_lm

        # Update for base model (0.6 GB per batch, 2x the VRAM)
        update_gpu_config_for_model_type(is_turbo=False)

        # Verify batch limits are reduced for base model
        updated_config = get_global_gpu_config()
        self.assertLessEqual(
            updated_config.max_batch_size_without_lm,
            initial_without_lm,
            "Base model should have same or lower batch limit without LM"
        )
        self.assertLessEqual(
            updated_config.max_batch_size_with_lm,
            initial_with_lm,
            "Base model should have same or lower batch limit with LM"
        )

    def test_turbo_model_allows_higher_batch_size(self):
        """Turbo model (no CFG) should support higher batch sizes."""
        # Start with base model config (lower limits)
        update_gpu_config_for_model_type(is_turbo=False)
        base_without_lm = get_global_gpu_config().max_batch_size_without_lm
        base_with_lm = get_global_gpu_config().max_batch_size_with_lm

        # Update for turbo model
        update_gpu_config_for_model_type(is_turbo=True)

        # Verify batch limits increase for turbo model
        turbo_config = get_global_gpu_config()
        self.assertGreaterEqual(
            turbo_config.max_batch_size_without_lm,
            base_without_lm,
            "Turbo model should have same or higher batch limit without LM"
        )
        self.assertGreaterEqual(
            turbo_config.max_batch_size_with_lm,
            base_with_lm,
            "Turbo model should have same or higher batch limit with LM"
        )

    def test_vram_per_batch_constants(self):
        """Verify VRAM per batch constants are correctly defined."""
        self.assertEqual(
            DIT_INFERENCE_VRAM_PER_BATCH["turbo"],
            0.3,
            "Turbo model should use 0.3 GB per batch"
        )
        self.assertEqual(
            DIT_INFERENCE_VRAM_PER_BATCH["base"],
            0.6,
            "Base model should use 0.6 GB per batch (2x turbo due to CFG)"
        )

    def test_batch_size_ratio_matches_vram_ratio(self):
        """Batch size reduction for base vs turbo should match VRAM per batch ratio."""
        # Update for turbo model
        update_gpu_config_for_model_type(is_turbo=True)
        turbo_config = get_global_gpu_config()
        turbo_batch = turbo_config.max_batch_size_without_lm

        # Update for base model
        update_gpu_config_for_model_type(is_turbo=False)
        base_config = get_global_gpu_config()
        base_batch = base_config.max_batch_size_without_lm

        # VRAM ratio is 2x (0.6 / 0.3), so batch size should be roughly half
        # Allow some slack due to rounding and safety margins
        expected_ratio = DIT_INFERENCE_VRAM_PER_BATCH["base"] / DIT_INFERENCE_VRAM_PER_BATCH["turbo"]
        actual_ratio = turbo_batch / max(base_batch, 1)  # Avoid division by zero

        self.assertGreaterEqual(
            actual_ratio,
            expected_ratio * 0.8,  # Within 20% tolerance
            f"Batch size ratio ({actual_ratio:.2f}) should be close to VRAM ratio ({expected_ratio:.2f})"
        )

    def test_preserves_other_config_fields(self):
        """Updating batch sizes should not modify other config fields."""
        initial_config = get_global_gpu_config()
        initial_tier = initial_config.tier
        initial_gpu_memory = initial_config.gpu_memory_gb
        initial_duration_with_lm = initial_config.max_duration_with_lm

        # Update for base model
        update_gpu_config_for_model_type(is_turbo=False)

        updated_config = get_global_gpu_config()
        self.assertEqual(updated_config.tier, initial_tier)
        self.assertEqual(updated_config.gpu_memory_gb, initial_gpu_memory)
        self.assertEqual(updated_config.max_duration_with_lm, initial_duration_with_lm)


if __name__ == "__main__":
    unittest.main()
