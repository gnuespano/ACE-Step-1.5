"""LoRA management mixin for AceStepHandler."""

import math
import os
from typing import Any, Dict

from loguru import logger
from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log


class LoraManagerMixin:
    """LoRA management behavior mixed into AceStepHandler.

    Expected host attributes:
    - model, device, dtype
    - _base_decoder
    - lora_loaded, use_lora, lora_scale
    """

    def _ensure_lora_registry(self) -> None:
        if not hasattr(self, "_lora_adapter_registry"):
            self._lora_adapter_registry = {}
        if not hasattr(self, "_lora_active_adapter"):
            self._lora_active_adapter = None
        if not hasattr(self, "_lora_scale_state"):
            self._lora_scale_state = {}
        if not hasattr(self, "_lora_last_scale_report"):
            self._lora_last_scale_report = {}

    def _debug_lora_registry_snapshot(self, max_targets_per_adapter: int = 20) -> Dict[str, Any]:
        """Return debugger-friendly snapshot of LoRA adapter registry."""
        self._ensure_lora_registry()
        adapters: Dict[str, Any] = {}
        for adapter_name, meta in self._lora_adapter_registry.items():
            targets = meta.get("targets", [])
            entries = []
            for t in targets[:max_targets_per_adapter]:
                module = t.get("module")
                entries.append(
                    {
                        "kind": t.get("kind"),
                        "module_name": t.get("module_name"),
                        "adapter": t.get("adapter"),
                        "module_class": module.__class__.__name__ if module is not None else None,
                    }
                )
            adapters[adapter_name] = {
                "path": meta.get("path"),
                "target_count": len(targets),
                "targets": entries,
                "truncated": len(targets) > max_targets_per_adapter,
            }
        return {
            "active_adapter": self._lora_active_adapter,
            "adapter_names": list(self._lora_adapter_registry.keys()),
            "adapters": adapters,
        }

    def _collect_adapter_names(self) -> list[str]:
        """Best-effort adapter name discovery across PEFT runtime variants."""
        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            return []

        def _extract_names(value) -> list[str]:
            names: list[str] = []

            def _append_name(v):
                if isinstance(v, str) and v and v not in names:
                    names.append(v)

            def _walk(v):
                if v is None:
                    return
                if isinstance(v, str):
                    _append_name(v)
                    return
                if isinstance(v, dict):
                    for k in v.keys():
                        _append_name(k)
                    return
                if isinstance(v, (list, tuple, set)):
                    for item in v:
                        _walk(item)
                    return
                if hasattr(v, "keys") and callable(v.keys):
                    try:
                        for k in v.keys():
                            _append_name(k)
                    except Exception:
                        pass
                if hasattr(v, "adapters"):
                    _walk(getattr(v, "adapters"))
                if hasattr(v, "adapter_names"):
                    _walk(getattr(v, "adapter_names"))
                if hasattr(v, "to_dict") and callable(v.to_dict):
                    try:
                        _walk(v.to_dict())
                    except Exception:
                        pass

            _walk(value)
            # Keep discovery order stable within each source group.
            return list(dict.fromkeys(names))

        ordered: list[str] = []
        source_groups: list[list[str]] = []

        # Preserve source priority: get_adapter_names > active_adapter > active_adapters > peft_config.
        if hasattr(decoder, "get_adapter_names") and callable(decoder.get_adapter_names):
            try:
                source_groups.append(_extract_names(decoder.get_adapter_names()))
            except Exception:
                pass

        if hasattr(decoder, "active_adapter"):
            try:
                source_groups.append(_extract_names(decoder.active_adapter))
            except Exception:
                pass

        if hasattr(decoder, "active_adapters"):
            try:
                active = decoder.active_adapters
                source_groups.append(_extract_names(active() if callable(active) else active))
            except Exception:
                pass

        if hasattr(decoder, "peft_config"):
            source_groups.append(_extract_names(getattr(decoder, "peft_config")))

        for group in source_groups:
            for name in group:
                if name not in ordered:
                    ordered.append(name)
        return ordered

    @staticmethod
    def _is_lora_like_module(name: str, module) -> bool:
        """Conservative LoRA module detection for mixed PEFT implementations."""
        name_l = name.lower()
        cls_l = module.__class__.__name__.lower()
        mod_l = module.__class__.__module__.lower()
        has_lora_signals = (
            "lora" in name_l
            or "lora" in cls_l
            or ("peft" in mod_l and "lora" in mod_l)
            or hasattr(module, "lora_A")
            or hasattr(module, "lora_B")
        )
        has_scaling_api = (
            hasattr(module, "scaling")
            or hasattr(module, "set_scale")
            or hasattr(module, "scale_layer")
        )
        return has_lora_signals and has_scaling_api

    @staticmethod
    def _read_adapter_value(value, adapter: str):
        """Read adapter-specific value from mapping-like or scalar containers."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get(adapter)
        if hasattr(value, "keys") and callable(value.keys):
            try:
                return value.get(adapter)
            except Exception:
                return None
        if isinstance(value, (int, float)):
            return value
        return None

    @staticmethod
    def _is_peft_factor_set_scale_module(module) -> bool:
        """Detect modules where set_scale(adapter, factor) semantics are expected."""
        return (
            hasattr(module, "set_scale")
            and hasattr(module, "lora_alpha")
            and hasattr(module, "r")
        )

    def _get_peft_initial_scale(self, module, adapter: str) -> float | None:
        """Return PEFT LoRA baseline scale (alpha/r or alpha/sqrt(r)) for adapter."""
        try:
            alpha = self._read_adapter_value(getattr(module, "lora_alpha", None), adapter)
            r_val = self._read_adapter_value(getattr(module, "r", None), adapter)
            if not isinstance(alpha, (int, float)) or not isinstance(r_val, (int, float)):
                return None
            if not r_val:
                return None
            use_rslora_raw = getattr(module, "use_rslora", False)
            if isinstance(use_rslora_raw, dict):
                use_rslora = bool(use_rslora_raw.get(adapter, False))
            else:
                use_rslora = bool(use_rslora_raw)
            return (alpha / math.sqrt(r_val)) if use_rslora else (alpha / r_val)
        except Exception as e:
            debug_log(
                lambda: f"Failed to compute initial scale (adapter={adapter}, err={e})",
                mode=DEBUG_MODEL_LOADING,
                prefix="lora",
            )
            return None

    def _rebuild_lora_registry(self, lora_path: str | None = None) -> tuple[int, list[str]]:
        """Build explicit adapter->target mapping used for deterministic scaling."""
        self._ensure_lora_registry()
        self._lora_adapter_registry = {}
        self._lora_scale_state = {}

        adapter_names = self._collect_adapter_names()
        if not adapter_names:
            adapter_names = ["default"]
        adapter_names = [a for a in adapter_names if isinstance(a, str) and a]
        if not adapter_names:
            adapter_names = ["default"]

        for adapter in adapter_names:
            self._lora_adapter_registry[adapter] = {
                "path": lora_path,
                "targets": [],
            }

        for module_name, module in self.model.decoder.named_modules():
            if not self._is_lora_like_module(module_name, module):
                continue

            # Path 1 (preferred): PEFT LoRA set_scale(adapter, factor).
            if self._is_peft_factor_set_scale_module(module):
                for adapter in adapter_names:
                    base_factor = None
                    scaling = getattr(module, "scaling", None)
                    current_scale = self._read_adapter_value(scaling, adapter)
                    initial_scale = self._get_peft_initial_scale(module, adapter)
                    if (
                        isinstance(current_scale, (int, float))
                        and isinstance(initial_scale, (int, float))
                        and initial_scale != 0
                    ):
                        base_factor = float(current_scale) / float(initial_scale)
                    self._lora_adapter_registry[adapter]["targets"].append(
                        {
                            "module": module,
                            "kind": "set_scale_factor",
                            "adapter": adapter,
                            "module_name": module_name,
                            "base_factor": base_factor,
                            "anchored": isinstance(base_factor, (int, float)),
                        }
                    )
                continue

            # Path 2: direct scaling dict keyed by adapter name.
            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                for adapter in adapter_names:
                    if adapter in module.scaling:
                        self._lora_adapter_registry[adapter]["targets"].append(
                            {
                                "module": module,
                                "kind": "scaling_dict",
                                "adapter": adapter,
                                "module_name": module_name,
                                "base_scale": module.scaling[adapter],
                            }
                        )
                continue

            # Path 3: unknown set_scale semantics; only use when we can anchor on observed base.
            if hasattr(module, "set_scale"):
                for adapter in adapter_names:
                    base_scale = self._read_adapter_value(getattr(module, "scaling", None), adapter)
                    self._lora_adapter_registry[adapter]["targets"].append(
                        {
                            "module": module,
                            "kind": "set_scale_unknown",
                            "adapter": adapter,
                            "module_name": module_name,
                            "base_scale": base_scale,
                        }
                    )
                continue

            # Path 4: adapter-agnostic scaling API (safe only for single-adapter).
            if hasattr(module, "scale_layer") and len(adapter_names) == 1:
                adapter = adapter_names[0]
                base_scale = self._read_adapter_value(getattr(module, "scaling", None), adapter)
                self._lora_adapter_registry[adapter]["targets"].append(
                    {
                        "module": module,
                        "kind": "scale_layer",
                        "module_name": module_name,
                        "base_scale": float(base_scale) if isinstance(base_scale, (int, float)) else None,
                    }
                )
                continue

            # Path 5: scalar scaling API (safe only for single-adapter).
            if hasattr(module, "scaling") and isinstance(module.scaling, (int, float)) and len(adapter_names) == 1:
                adapter = adapter_names[0]
                self._lora_adapter_registry[adapter]["targets"].append(
                    {
                        "module": module,
                        "kind": "scaling_scalar",
                        "module_name": module_name,
                        "base_scale": float(module.scaling),
                    }
                )

        total_targets = sum(len(meta["targets"]) for meta in self._lora_adapter_registry.values())

        if self._lora_active_adapter not in self._lora_adapter_registry:
            self._lora_active_adapter = next(iter(self._lora_adapter_registry.keys()), None)

        return total_targets, list(self._lora_adapter_registry.keys())

    def _apply_scale_to_adapter(self, adapter_name: str, scale: float) -> int:
        """Apply scale to registered targets for one adapter."""
        self._ensure_lora_registry()
        meta = self._lora_adapter_registry.get(adapter_name)
        if not meta:
            self._lora_last_scale_report = {
                "adapter": adapter_name,
                "modified_total": 0,
                "modified_by_kind": {},
                "skipped_by_kind": {"no_registry": 1},
            }
            return 0

        modified = 0
        modified_by_kind: Dict[str, int] = {}
        skipped_by_kind: Dict[str, int] = {}
        for target in meta.get("targets", []):
            module = target.get("module")
            kind = target.get("kind")
            if module is None:
                skipped_by_kind[kind] = skipped_by_kind.get(kind, 0) + 1
                continue

            try:
                if kind == "scaling_dict":
                    adapter = target.get("adapter")
                    if adapter not in module.scaling:
                        skipped_by_kind[kind] = skipped_by_kind.get(kind, 0) + 1
                        continue
                    base_scale = target.get("base_scale", module.scaling[adapter])
                    module.scaling[adapter] = base_scale * scale
                    modified += 1
                    modified_by_kind[kind] = modified_by_kind.get(kind, 0) + 1
                elif kind == "set_scale_factor":
                    base_factor = target.get("base_factor", None)
                    anchored = bool(target.get("anchored", False))
                    if anchored and isinstance(base_factor, (int, float)):
                        factor = base_factor * scale
                        module.set_scale(adapter_name, factor)
                        modified += 1
                        modified_by_kind[kind] = modified_by_kind.get(kind, 0) + 1
                    else:
                        skipped_by_kind["set_scale_factor_unanchored"] = skipped_by_kind.get(
                            "set_scale_factor_unanchored", 0
                        ) + 1
                        logger.warning(
                            f"Skipping set_scale_factor target without anchor "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
                elif kind == "set_scale_unknown":
                    # Unknown third-party semantics: anchor to observed base when available.
                    base_scale = target.get("base_scale", None)
                    if isinstance(base_scale, (int, float)):
                        module.set_scale(adapter_name, base_scale * scale)
                        modified += 1
                        modified_by_kind[kind] = modified_by_kind.get(kind, 0) + 1
                    else:
                        skipped_by_kind[kind] = skipped_by_kind.get(kind, 0) + 1
                        logger.warning(
                            f"Skipping set_scale target with unknown semantics and no base "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
                        debug_log(
                            lambda: (
                                f"Skipped unanchored set_scale target "
                                f"(adapter={adapter_name}, module={target.get('module_name')})"
                            ),
                            mode=DEBUG_MODEL_LOADING,
                            prefix="lora",
                        )
                elif kind == "scale_layer":
                    base_scale = target.get("base_scale", None)
                    desired = (base_scale * scale) if isinstance(base_scale, (int, float)) else scale
                    if hasattr(module, "unscale_layer"):
                        kind_key = kind
                        if base_scale is None:
                            # Applied with fallback baseline: report separately from "skipped".
                            kind_key = "scale_layer_fallback"
                        # For PEFT LoRA layers this resets to initial scale deterministically.
                        module.unscale_layer()
                        module.scale_layer(desired)
                        modified += 1
                        modified_by_kind[kind_key] = modified_by_kind.get(kind_key, 0) + 1
                    else:
                        if base_scale is None:
                            skipped_by_kind["scale_layer_unanchored"] = skipped_by_kind.get("scale_layer_unanchored", 0) + 1
                            logger.warning(
                                f"Skipping unanchored scale_layer target without unscale_layer "
                                f"(adapter={adapter_name}, module={target.get('module_name')})"
                            )
                            continue
                        state_key = (id(module), kind, adapter_name)
                        prev = self._lora_scale_state.get(state_key)
                        if isinstance(prev, (int, float)) and prev > 0:
                            module.scale_layer(desired / prev)
                        else:
                            module.scale_layer(desired)
                        self._lora_scale_state[state_key] = float(desired)
                        modified += 1
                        modified_by_kind[kind] = modified_by_kind.get(kind, 0) + 1
                elif kind == "scaling_scalar":
                    base_scale = target.get("base_scale", float(module.scaling))
                    module.scaling = base_scale * scale
                    modified += 1
                    modified_by_kind[kind] = modified_by_kind.get(kind, 0) + 1
            except Exception:
                skipped_by_kind[kind] = skipped_by_kind.get(kind, 0) + 1
                continue

        self._lora_last_scale_report = {
            "adapter": adapter_name,
            "modified_total": modified,
            "modified_by_kind": modified_by_kind,
            "skipped_by_kind": skipped_by_kind,
        }
        return modified

    def load_lora(self, lora_path: str) -> str:
        """Load LoRA adapter into the decoder."""
        if self.model is None:
            return "❌ Model not initialized. Please initialize service first."

        # Check if model is quantized - LoRA loading on quantized models is not supported
        # due to incompatibility between PEFT and torchao (missing get_apply_tensor_subclass argument)
        if self.quantization is not None:
            return (
                f"❌ LoRA loading is not supported on quantized models. "
                f"Current quantization: {self.quantization}. "
                "Please re-initialize the service with quantization disabled, then try loading the LoRA adapter again."
            )

        if not lora_path or not lora_path.strip():
            return "❌ Please provide a LoRA path."

        lora_path = lora_path.strip()

        # Check if path exists
        if not os.path.exists(lora_path):
            return f"❌ LoRA path not found: {lora_path}"

        # Check if it's a valid PEFT adapter directory
        config_file = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_file):
            return f"❌ Invalid LoRA adapter: adapter_config.json not found in {lora_path}"

        try:
            from peft import PeftModel, PeftConfig
        except ImportError:
            return "❌ PEFT library not installed. Please install with: pip install peft"

        try:
            import copy
            # Backup base decoder if not already backed up
            if self._base_decoder is None:
                self._base_decoder = copy.deepcopy(self.model.decoder)
                logger.info("Base decoder backed up")
            else:
                # Restore base decoder before loading new LoRA
                self.model.decoder = copy.deepcopy(self._base_decoder)
                logger.info("Restored base decoder before loading new LoRA")

            # Load PEFT adapter
            logger.info(f"Loading LoRA adapter from {lora_path}")
            self.model.decoder = PeftModel.from_pretrained(
                self.model.decoder,
                lora_path,
                is_trainable=False,
            )
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()

            self.lora_loaded = True
            self.use_lora = True  # Enable LoRA by default after loading
            self._ensure_lora_registry()
            self._lora_active_adapter = None
            target_count, adapters = self._rebuild_lora_registry(lora_path=lora_path)

            logger.info(
                f"LoRA adapter loaded successfully from {lora_path} "
                f"(adapters={adapters}, targets={target_count})"
            )
            debug_log(
                lambda: f"LoRA registry snapshot: {self._debug_lora_registry_snapshot()}",
                mode=DEBUG_MODEL_LOADING,
                prefix="lora",
            )
            return f"✅ LoRA loaded from {lora_path}"

        except Exception as e:
            logger.exception("Failed to load LoRA adapter")
            return f"❌ Failed to load LoRA: {str(e)}"

    def unload_lora(self) -> str:
        """Unload LoRA adapter and restore base decoder."""
        if not self.lora_loaded:
            return "⚠️ No LoRA adapter loaded."

        if self._base_decoder is None:
            return "❌ Base decoder backup not found. Cannot restore."

        try:
            import copy
            # Restore base decoder
            self.model.decoder = copy.deepcopy(self._base_decoder)
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()

            self.lora_loaded = False
            self.use_lora = False
            self.lora_scale = 1.0  # Reset scale to default
            self._ensure_lora_registry()
            self._lora_adapter_registry = {}
            self._lora_active_adapter = None
            self._lora_scale_state = {}

            logger.info("LoRA unloaded, base decoder restored")
            return "✅ LoRA unloaded, using base model"

        except Exception as e:
            logger.exception("Failed to unload LoRA")
            return f"❌ Failed to unload LoRA: {str(e)}"

    def set_use_lora(self, use_lora: bool) -> str:
        """Toggle LoRA usage for inference."""
        if use_lora and not self.lora_loaded:
            return "❌ No LoRA adapter loaded. Please load a LoRA first."

        self.use_lora = use_lora

        # Use PEFT's enable/disable methods if available
        if self.lora_loaded and hasattr(self.model.decoder, "disable_adapter_layers"):
            try:
                if use_lora:
                    if self._lora_active_adapter and hasattr(self.model.decoder, "set_adapter"):
                        try:
                            self.model.decoder.set_adapter(self._lora_active_adapter)
                        except Exception:
                            pass
                    self.model.decoder.enable_adapter_layers()
                    logger.info("LoRA adapter enabled")
                    # Apply current scale when enabling LoRA
                    if self.lora_scale != 1.0:
                        self.set_lora_scale(self.lora_scale)
                else:
                    self.model.decoder.disable_adapter_layers()
                    logger.info("LoRA adapter disabled")
            except Exception as e:
                logger.warning(f"Could not toggle adapter layers: {e}")

        status = "enabled" if use_lora else "disabled"
        return f"✅ LoRA {status}"

    def set_lora_scale(self, scale: float) -> str:
        """Set LoRA adapter scale/weight (0-1 range)."""
        if not self.lora_loaded:
            return "⚠️ No LoRA loaded"

        # Clamp scale to 0-1 range
        self.lora_scale = max(0.0, min(1.0, scale))

        # Only apply scaling if LoRA is enabled
        if not self.use_lora:
            logger.info(f"LoRA scale set to {self.lora_scale:.2f} (will apply when LoRA is enabled)")
            return f"✅ LoRA scale: {self.lora_scale:.2f} (LoRA disabled)"

        try:
            if not getattr(self, "_lora_adapter_registry", None):
                self._rebuild_lora_registry()

            active_adapter = self._lora_active_adapter
            if active_adapter is None and self._lora_adapter_registry:
                active_adapter = next(iter(self._lora_adapter_registry.keys()))
                self._lora_active_adapter = active_adapter

            debug_log(
                lambda: (
                    f"LoRA scale request: slider={self.lora_scale:.3f} "
                    f"active_adapter={active_adapter} adapters={list(self._lora_adapter_registry.keys())}"
                ),
                mode=DEBUG_MODEL_LOADING,
                prefix="lora",
            )

            modified_count = self._apply_scale_to_adapter(active_adapter, self.lora_scale) if active_adapter else 0
            report = getattr(self, "_lora_last_scale_report", {})

            if modified_count > 0 and active_adapter:
                logger.info(
                    f"LoRA scale set to {self.lora_scale:.2f} "
                    f"(adapter={active_adapter}, modified={modified_count}, "
                    f"by_kind={report.get('modified_by_kind', {})}, skipped={report.get('skipped_by_kind', {})})"
                )
                skipped_total = sum(report.get("skipped_by_kind", {}).values())
                if skipped_total > 0:
                    return f"✅ LoRA scale: {self.lora_scale:.2f} (skipped {skipped_total} targets)"
                return f"✅ LoRA scale: {self.lora_scale:.2f}"
            else:
                logger.warning(
                    f"No registered LoRA scaling targets found for active adapter "
                    f"(skipped={report.get('skipped_by_kind', {})})"
                )
                return f"⚠️ Scale set to {self.lora_scale:.2f} (no modules found)"
        except Exception as e:
            logger.warning(f"Could not set LoRA scale: {e}")
            return f"⚠️ Scale set to {self.lora_scale:.2f} (partial)"

    def set_active_lora_adapter(self, adapter_name: str) -> str:
        """Set the active LoRA adapter for scaling/inference.

        This is backward compatible with single-adapter UI and is forward-compatible
        for future multi-LoRA controls.
        """
        self._ensure_lora_registry()
        if adapter_name not in self._lora_adapter_registry:
            return f"❌ Unknown adapter: {adapter_name}"
        self._lora_active_adapter = adapter_name
        debug_log(f"Selected active LoRA adapter: {adapter_name}", mode=DEBUG_MODEL_LOADING, prefix="lora")
        if self.model is not None and hasattr(self.model, "decoder") and hasattr(self.model.decoder, "set_adapter"):
            try:
                self.model.decoder.set_adapter(adapter_name)
            except Exception:
                pass
        return f"✅ Active LoRA adapter: {adapter_name}"

    def get_lora_status(self) -> Dict[str, Any]:
        """Get current LoRA status."""
        self._ensure_lora_registry()
        return {
            "loaded": self.lora_loaded,
            "active": self.use_lora,
            "scale": self.lora_scale,
            "active_adapter": self._lora_active_adapter,
            "adapters": list(self._lora_adapter_registry.keys()),
        }
