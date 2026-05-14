"""
Generic LoRA mixin for models built on :class:`~nunchaku.models.linear.SVDQW4A4Linear`.

A transformer model class only needs to:

1. Inherit :class:`SVDQLoRAMixin`.
2. Implement :meth:`_lora_key_map` returning a configuration dict.
3. Call :meth:`_init_lora_state` at the end of ``from_pretrained``.

Then it automatically gains :meth:`update_lora_params`, :meth:`set_lora_strength`,
and :meth:`reset_lora`.
"""

from __future__ import annotations

import logging
from typing import Any, Generator

import torch
import torch.nn as nn

from ...utils import load_state_dict_in_safetensors
from ..flux.nunchaku_converter import pack_lowrank_weight, unpack_lowrank_weight

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key-map schema
# ---------------------------------------------------------------------------
#
# _lora_key_map() must return a dict with the following structure:
#
#   {
#       "block_prefixes": ["layers", "noise_refiner", ...],
#       "quantized_targets": {
#           "<module_suffix>": {
#               "lora_keys": [...],      # diffusers LoRA key suffixes
#               "type": "linear" | "fused_qkv" | "fused_split",
#               "param_prefix": "...",   # (optional) override for actual parameter path
#               "applies_to": [...],     # (optional) restrict to these block_prefixes;
#                                        #   omit to apply to all block_prefixes
#               # fused_split only:
#               "targets": [...],        # list of target module suffixes to split into
#               "split_dim": "output" | "input",  # which dim of lora_B/A to split
#           },
#           ...
#       },
#       "unquantized_targets": {
#           "<module_suffix>": {
#               "lora_keys": [...],
#               "type": "linear",
#           },
#           ...
#       },
#   }
#


class SVDQLoRAMixin:
    """Mixin that adds LoRA lifecycle management to any SVDQ-based transformer."""

    # ── subclass contract ──────────────────────────────────────────────────

    def _lora_key_map(self) -> dict[str, Any]:
        """Return the LoRA key-mapping configuration for this model.

        See the module-level docstring for the expected schema.
        """
        raise NotImplementedError

    def _pre_convert_lora_sd(
        self, lora_sd: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Model-specific LoRA key pre-processing before generic conversion.

        Subclasses can override this to remap non-standard key names
        (e.g. ZImage ``feed_forward.w1`` → ``feed_forward.net.0.proj``).
        Called after generic format normalisation but before key matching.
        """
        return lora_sd

    def _unwrap_quantized_module(self, module: nn.Module):
        """Return the underlying SVDQW4A4Linear if *module* wraps one, else *module* itself.

        Subclasses should override when quantized linears are wrapped
        (e.g. ``ZImageTurboFFNDownGuard``).
        """
        return module

    # ── public API ─────────────────────────────────────────────────────────

    def update_lora_params(
        self,
        path_or_state_dict: str | dict[str, torch.Tensor],
        strength: float = 1.0,
    ) -> None:
        """Load and apply LoRA weights.

        Parameters
        ----------
        path_or_state_dict : str or dict
            Path to a ``.safetensors`` file (local or HuggingFace), or an
            already-loaded state dict (e.g. output of :func:`compose_lora`).
        strength : float
            Scaling factor applied to the LoRA contribution.
        """
        if strength < 0:
            raise ValueError(f"LoRA strength must be non-negative, got {strength}")

        if isinstance(path_or_state_dict, dict):
            lora_sd = {k: v for k, v in path_or_state_dict.items()}
        else:
            lora_sd = load_state_dict_in_safetensors(path_or_state_dict)

        if not lora_sd:
            raise ValueError("LoRA state dict is empty")

        self._reset_quantized_loras()
        self._reset_unquantized_loras()

        quantized, unquantized = self._convert_lora_keys(lora_sd)
        self._cached_quantized_loras = quantized
        self._cached_unquantized_loras = unquantized
        self._lora_strength = strength

        if quantized:
            self._apply_quantized_loras(quantized, strength)
        if unquantized:
            self._apply_unquantized_loras(unquantized, strength)

        n_q = len(quantized) // 2
        n_uq = len(unquantized) // 2
        logger.info(
            "LoRA applied: strength=%s, %d quantized layers, %d unquantized layers",
            strength, n_q, n_uq,
        )

    def set_lora_strength(self, strength: float = 1.0) -> None:
        """Adjust LoRA strength without reloading weights.

        Parameters
        ----------
        strength : float
            New scaling factor.
        """
        if strength < 0:
            raise ValueError(f"LoRA strength must be non-negative, got {strength}")

        if not hasattr(self, "_cached_quantized_loras"):
            logger.warning("No LoRA loaded; call update_lora_params first.")
            return

        self._reset_quantized_loras()
        self._reset_unquantized_loras()

        self._lora_strength = strength

        if self._cached_quantized_loras:
            self._apply_quantized_loras(self._cached_quantized_loras, strength)
        if self._cached_unquantized_loras:
            self._apply_unquantized_loras(self._cached_unquantized_loras, strength)

        logger.info("LoRA strength set to %s", strength)

    def reset_lora(self) -> None:
        """Remove all LoRA effects and restore original weights."""
        self._reset_quantized_loras()
        self._reset_unquantized_loras()

        self._cached_quantized_loras = {}
        self._cached_unquantized_loras = {}
        self._lora_strength = 1.0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("LoRA reset to original state")

    # ── shared iteration helpers ───────────────────────────────────────────

    def _iter_quantized_targets(
        self,
    ) -> Generator[tuple[str, int, str, dict], None, None]:
        """Yield ``(block_prefix, idx, target_suffix, cfg)`` for all matching quantized targets."""
        key_map = self._lora_key_map()
        for block_prefix in key_map["block_prefixes"]:
            block_list = getattr(self, block_prefix, None)
            if block_list is None:
                continue
            for idx in range(len(block_list)):
                for target_suffix, cfg in key_map["quantized_targets"].items():
                    applies_to = cfg.get("applies_to")
                    if applies_to is not None and block_prefix not in applies_to:
                        continue
                    yield block_prefix, idx, target_suffix, cfg

    def _iter_unquantized_targets(
        self,
    ) -> Generator[tuple[str, int, str, dict], None, None]:
        """Yield ``(block_prefix, idx, target_suffix, cfg)`` for all matching unquantized targets."""
        key_map = self._lora_key_map()
        for block_prefix in key_map["block_prefixes"]:
            block_list = getattr(self, block_prefix, None)
            if block_list is None:
                continue
            for idx in range(len(block_list)):
                for target_suffix, cfg in key_map.get("unquantized_targets", {}).items():
                    yield block_prefix, idx, target_suffix, cfg

    @staticmethod
    def _resolve_fused_attrs(
        model: Any, block_prefix: str, idx: int, param_prefix: str
    ) -> tuple[Any, str, str, str] | None:
        """Resolve the parent module and attribute names for a fused-module ``param_prefix``.

        Returns ``(parent_module, full_prefix, down_attr, up_attr)`` or
        ``None`` if the parent module cannot be found.
        """
        parent_path = f"{block_prefix}.{idx}.{param_prefix.rsplit('.', 1)[0]}"
        module = _get_nested_attr(model, parent_path)
        if module is None:
            return None
        full_prefix = f"{block_prefix}.{idx}.{param_prefix}"
        attr_base = param_prefix.rsplit(".", 1)[-1] if "." in param_prefix else param_prefix
        return module, full_prefix, f"{attr_base}proj_down", f"{attr_base}proj_up"

    # ── initialisation (call from from_pretrained) ─────────────────────────

    def _init_lora_state(self) -> None:
        """Snapshot current weights so LoRA can be applied / reset later."""
        from ...models.linear import SVDQW4A4Linear

        self._quantized_part_sd: dict[str, torch.Tensor] = {}
        self._quantized_part_ranks: dict[str, int] = {}
        self._unquantized_part_sd: dict[str, torch.Tensor] = {}
        self._cached_quantized_loras: dict[str, torch.Tensor] = {}
        self._cached_unquantized_loras: dict[str, torch.Tensor] = {}
        self._lora_strength = 1.0

        for block_prefix, idx, target_suffix, cfg in self._iter_quantized_targets():
            param_prefix = cfg.get("param_prefix")
            if param_prefix is not None:
                resolved = self._resolve_fused_attrs(self, block_prefix, idx, param_prefix)
                if resolved is None:
                    continue
                module, full_prefix, down_attr, up_attr = resolved
                down_param = getattr(module, down_attr, None)
                up_param = getattr(module, up_attr, None)
                if down_param is not None and up_param is not None:
                    key_down = f"{full_prefix}proj_down"
                    key_up = f"{full_prefix}proj_up"
                    if key_down not in self._quantized_part_sd:
                        self._quantized_part_sd[key_down] = down_param.data.clone()
                        self._quantized_part_sd[key_up] = up_param.data.clone()
                        self._quantized_part_ranks[full_prefix.rstrip(".")] = down_param.shape[1]
            else:
                full_name = f"{block_prefix}.{idx}.{target_suffix}"
                raw_module = _get_nested_attr(self, full_name)
                if raw_module is None:
                    continue
                module = self._unwrap_quantized_module(raw_module)
                if not isinstance(module, SVDQW4A4Linear):
                    continue
                if f"{full_name}.proj_down" not in self._quantized_part_sd:
                    self._quantized_part_sd[f"{full_name}.proj_down"] = module.proj_down.data.clone()
                    self._quantized_part_sd[f"{full_name}.proj_up"] = module.proj_up.data.clone()
                    self._quantized_part_ranks[full_name] = module.rank

        for block_prefix, idx, target_suffix, _cfg in self._iter_unquantized_targets():
            full_name = f"{block_prefix}.{idx}.{target_suffix}"
            module = _get_nested_attr(self, full_name)
            if module is None:
                continue
            if isinstance(module, nn.Linear):
                self._unquantized_part_sd[f"{full_name}.weight"] = module.weight.data.clone()
                if module.bias is not None:
                    self._unquantized_part_sd[f"{full_name}.bias"] = module.bias.data.clone()

        logger.info(
            "_init_lora_state: %d quantized modules snapshotted, %d unquantized params snapshotted",
            len(self._quantized_part_sd) // 2,
            len(self._unquantized_part_sd),
        )

    # ── format conversion ──────────────────────────────────────────────────

    def _convert_lora_keys(
        self, lora_sd: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Convert diffusers-format LoRA keys to internal quantized/unquantized dicts.

        Returns ``(quantized_loras, unquantized_loras)`` where keys are
        ``"<block_prefix>.<idx>.<target_suffix>.proj_down"`` (and ``proj_up``)
        for quantized, or ``"<full_name>.lora_A.weight"`` / ``lora_B`` for unquantized.
        """
        key_map = self._lora_key_map()
        lora_sd = _normalize_lora_format(lora_sd)
        lora_sd = self._pre_convert_lora_sd(lora_sd)

        quantized: dict[str, torch.Tensor] = {}
        unquantized: dict[str, torch.Tensor] = {}
        matched_keys: set[str] = set()

        for block_prefix in key_map["block_prefixes"]:
            block_list = getattr(self, block_prefix, None)
            if block_list is None:
                continue
            for idx in range(len(block_list)):
                block_key = f"{block_prefix}.{idx}"

                for target_suffix, cfg in key_map["quantized_targets"].items():
                    applies_to = cfg.get("applies_to")
                    if applies_to is not None and block_prefix not in applies_to:
                        continue
                    lora_keys = cfg["lora_keys"]
                    target_type = cfg["type"]

                    if target_type == "fused_qkv":
                        for lk in lora_keys:
                            for sfx in (".lora_A.weight", ".lora_B.weight"):
                                k = f"{block_key}.{lk}{sfx}"
                                if k in lora_sd:
                                    matched_keys.add(k)
                        self._convert_fused_qkv(
                            lora_sd, quantized, block_key, target_suffix, lora_keys, cfg
                        )
                    elif target_type == "fused_split":
                        for lk in lora_keys:
                            for sfx in (".lora_A.weight", ".lora_B.weight"):
                                k = f"{block_key}.{lk}{sfx}"
                                if k in lora_sd:
                                    matched_keys.add(k)
                        self._convert_fused_split(
                            lora_sd, quantized, block_key, lora_keys, cfg
                        )
                    elif target_type == "linear":
                        assert len(lora_keys) == 1
                        lora_key = lora_keys[0]
                        a_key = f"{block_key}.{lora_key}.lora_A.weight"
                        b_key = f"{block_key}.{lora_key}.lora_B.weight"
                        if a_key in lora_sd and b_key in lora_sd:
                            matched_keys.update([a_key, b_key])
                            lora_a = lora_sd[a_key]  # (rank, in_features)
                            lora_b = lora_sd[b_key]  # (out_features, rank)
                            out_key = f"{block_key}.{target_suffix}"
                            quantized[f"{out_key}.proj_down"] = lora_a.contiguous()
                            quantized[f"{out_key}.proj_up"] = lora_b.contiguous()

                for target_suffix, cfg in key_map.get("unquantized_targets", {}).items():
                    lora_keys = cfg["lora_keys"]
                    for lora_key in lora_keys:
                        a_key = f"{block_key}.{lora_key}.lora_A.weight"
                        b_key = f"{block_key}.{lora_key}.lora_B.weight"
                        if a_key in lora_sd and b_key in lora_sd:
                            matched_keys.update([a_key, b_key])
                            out_prefix = f"{block_key}.{target_suffix}"
                            unquantized[f"{out_prefix}.lora_A.weight"] = lora_sd[a_key]
                            unquantized[f"{out_prefix}.lora_B.weight"] = lora_sd[b_key]

        missed_keys = sorted(k for k in lora_sd if k not in matched_keys)
        if missed_keys:
            logger.warning(
                "LoRA key conversion: %d keys matched, %d keys missed.",
                len(matched_keys), len(missed_keys),
            )
            for k in missed_keys[:20]:
                logger.warning("  MISSED: %s", k)
            if len(missed_keys) > 20:
                logger.warning("  ... and %d more", len(missed_keys) - 20)
        else:
            logger.info("LoRA key conversion: all %d keys matched.", len(matched_keys))

        return quantized, unquantized

    def _convert_fused_qkv(
        self,
        lora_sd: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        block_key: str,
        target_suffix: str,
        lora_keys: list[str],
        cfg: dict,
    ) -> None:
        """Fuse separate Q/K/V LoRA into a single block-diagonal QKV LoRA."""
        parts_a, parts_b = [], []
        for lora_key in lora_keys:
            a = lora_sd.get(f"{block_key}.{lora_key}.lora_A.weight")
            b = lora_sd.get(f"{block_key}.{lora_key}.lora_B.weight")
            if a is None or b is None:
                return
            parts_a.append(a)
            parts_b.append(b)

        ranks = [a.shape[0] for a in parts_a]
        if len(set(ranks)) > 1:
            max_rank = max(ranks)
            parts_a = [torch.nn.functional.pad(a, (0, 0, 0, max_rank - a.shape[0])) for a in parts_a]
            parts_b = [torch.nn.functional.pad(b, (0, max_rank - b.shape[1])) for b in parts_b]
            ranks = [max_rank] * len(ranks)

        rank = ranks[0]

        if all(torch.equal(parts_a[0], a) for a in parts_a[1:]):
            fused_down = parts_a[0].contiguous()
            fused_up = torch.cat(parts_b, dim=0).contiguous()
        else:
            fused_down = torch.cat(parts_a, dim=0).contiguous()
            total_out = sum(b.shape[0] for b in parts_b)
            total_rank = rank * len(parts_b)
            fused_up = torch.zeros(total_out, total_rank, dtype=parts_b[0].dtype, device=parts_b[0].device)
            row_off, col_off = 0, 0
            for b in parts_b:
                fused_up[row_off : row_off + b.shape[0], col_off : col_off + rank] = b
                row_off += b.shape[0]
                col_off += rank
            fused_up = fused_up.contiguous()

        out_key = f"{block_key}.{target_suffix}"
        out[f"{out_key}.proj_down"] = fused_down
        out[f"{out_key}.proj_up"] = fused_up

    def _convert_fused_split(
        self,
        lora_sd: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        block_key: str,
        lora_keys: list[str],
        cfg: dict,
    ) -> None:
        """Split a single fused LoRA into multiple target modules.

        Used when diffusers trains LoRA on a fused linear (e.g.
        ``to_qkv_mlp_proj``) but Nunchaku splits it into separate modules
        (e.g. ``qkv_proj`` + ``mlp_fc1``).

        ``cfg["split_dim"]`` determines how to partition:
        - ``"output"``: split ``lora_B`` along dim-0 (output features),
          share ``lora_A`` across all targets.
        - ``"input"``: split ``lora_A`` along dim-1 (input features),
          share ``lora_B`` across all targets.

        Dimensions are inferred from the target SVDQW4A4Linear modules at
        runtime via ``module.out_features`` / ``module.in_features``.
        """
        from ...models.linear import SVDQW4A4Linear

        assert len(lora_keys) == 1
        lora_key = lora_keys[0]
        a_key = f"{block_key}.{lora_key}.lora_A.weight"
        b_key = f"{block_key}.{lora_key}.lora_B.weight"
        if a_key not in lora_sd or b_key not in lora_sd:
            return

        lora_a = lora_sd[a_key]
        lora_b = lora_sd[b_key]

        targets = cfg["targets"]
        split_dim = cfg["split_dim"]

        if split_dim == "output":
            dims = []
            for t in targets:
                module = _get_nested_attr(self, f"{block_key}.{t}")
                if module is None or not isinstance(module, SVDQW4A4Linear):
                    return
                dims.append(module.out_features)
            parts_b = lora_b.split(dims, dim=0)
            for t, part_b in zip(targets, parts_b):
                out_key = f"{block_key}.{t}"
                out[f"{out_key}.proj_down"] = lora_a.contiguous()
                out[f"{out_key}.proj_up"] = part_b.contiguous()
        elif split_dim == "input":
            dims = []
            for t in targets:
                module = _get_nested_attr(self, f"{block_key}.{t}")
                if module is None or not isinstance(module, SVDQW4A4Linear):
                    return
                dims.append(module.in_features)
            parts_a = lora_a.split(dims, dim=1)
            for t, part_a in zip(targets, parts_a):
                out_key = f"{block_key}.{t}"
                out[f"{out_key}.proj_down"] = part_a.contiguous()
                out[f"{out_key}.proj_up"] = lora_b.contiguous()
        else:
            raise ValueError(f"Unknown split_dim: {split_dim!r}")

    # ── quantized weight operations ────────────────────────────────────────

    def _apply_quantized_loras(
        self, loras: dict[str, torch.Tensor], strength: float
    ) -> None:
        """Fuse external LoRA low-rank weights into SVDQW4A4Linear proj_down/proj_up."""
        from ...models.linear import SVDQW4A4Linear

        applied_count = 0

        for block_prefix, idx, target_suffix, cfg in self._iter_quantized_targets():
            out_key = f"{block_prefix}.{idx}.{target_suffix}"
            down_key = f"{out_key}.proj_down"
            up_key = f"{out_key}.proj_up"
            if down_key not in loras:
                continue

            extra_down = loras[down_key]
            extra_up = loras[up_key] * strength

            param_prefix = cfg.get("param_prefix")
            if param_prefix is not None:
                resolved = self._resolve_fused_attrs(self, block_prefix, idx, param_prefix)
                if resolved is None:
                    continue
                module, _full_prefix, down_attr, up_attr = resolved
                orig_packed_down = getattr(module, down_attr)
                orig_packed_up = getattr(module, up_attr)
                packed_down, packed_up, _new_rank = _fuse_lowrank_pair(
                    orig_packed_down, orig_packed_up, extra_down, extra_up,
                )
                setattr(module, down_attr, nn.Parameter(packed_down, requires_grad=False))
                setattr(module, up_attr, nn.Parameter(packed_up, requires_grad=False))
                applied_count += 1
            else:
                raw_module = _get_nested_attr(self, out_key)
                if raw_module is None:
                    continue
                module = self._unwrap_quantized_module(raw_module)
                if not isinstance(module, SVDQW4A4Linear):
                    continue
                packed_down, packed_up, new_rank = _fuse_lowrank_pair(
                    module.proj_down, module.proj_up, extra_down, extra_up,
                )
                module.proj_down = nn.Parameter(packed_down, requires_grad=False)
                module.proj_up = nn.Parameter(packed_up, requires_grad=False)
                module.rank = new_rank
                applied_count += 1

        logger.debug(
            "_apply_quantized_loras: %d modules fused (from %d cached pairs)",
            applied_count, len(loras) // 2,
        )

    def _reset_quantized_loras(self) -> None:
        """Restore all quantized layers to their original proj_down/proj_up."""
        from ...models.linear import SVDQW4A4Linear

        if not hasattr(self, "_quantized_part_sd") or not self._quantized_part_sd:
            return

        for block_prefix, idx, target_suffix, cfg in self._iter_quantized_targets():
            param_prefix = cfg.get("param_prefix")
            if param_prefix is not None:
                resolved = self._resolve_fused_attrs(self, block_prefix, idx, param_prefix)
                if resolved is None:
                    continue
                module, full_prefix, down_attr, up_attr = resolved
                down_key = f"{full_prefix}proj_down"
                up_key = f"{full_prefix}proj_up"
                if down_key not in self._quantized_part_sd:
                    continue
                device = getattr(module, down_attr).device
                setattr(
                    module, down_attr,
                    nn.Parameter(self._quantized_part_sd[down_key].clone().to(device=device), requires_grad=False),
                )
                setattr(
                    module, up_attr,
                    nn.Parameter(self._quantized_part_sd[up_key].clone().to(device=device), requires_grad=False),
                )
            else:
                full_name = f"{block_prefix}.{idx}.{target_suffix}"
                down_key = f"{full_name}.proj_down"
                up_key = f"{full_name}.proj_up"
                if down_key not in self._quantized_part_sd:
                    continue
                raw_module = _get_nested_attr(self, full_name)
                if raw_module is None:
                    continue
                module = self._unwrap_quantized_module(raw_module)
                if not isinstance(module, SVDQW4A4Linear):
                    continue
                orig_rank = self._quantized_part_ranks.get(full_name, module.rank)
                device = module.proj_down.device
                module.proj_down = nn.Parameter(
                    self._quantized_part_sd[down_key].clone().to(device=device), requires_grad=False,
                )
                module.proj_up = nn.Parameter(
                    self._quantized_part_sd[up_key].clone().to(device=device), requires_grad=False,
                )
                module.rank = orig_rank

    # ── unquantized weight operations ──────────────────────────────────────

    def _apply_unquantized_loras(
        self, loras: dict[str, torch.Tensor], strength: float
    ) -> None:
        """Apply LoRA to unquantized linear layers (e.g. adaLN_modulation)."""
        if not self._unquantized_part_sd:
            return

        for k, v in self._unquantized_part_sd.items():
            if k.endswith(".weight"):
                base_key = k.rsplit(".weight", 1)[0]
                a_key = f"{base_key}.lora_A.weight"
                b_key = f"{base_key}.lora_B.weight"
                if a_key in loras and b_key in loras:
                    param = _get_nested_param(self, k)
                    if param is None:
                        continue
                    lora_a = loras[a_key].to(device=param.device, dtype=param.dtype)
                    lora_b = loras[b_key].to(device=param.device, dtype=param.dtype)
                    orig = v.to(device=param.device, dtype=param.dtype)
                    param.data.copy_(orig + strength * (lora_b @ lora_a))

    def _reset_unquantized_loras(self) -> None:
        """Restore unquantized layers to their original weights."""
        if not hasattr(self, "_unquantized_part_sd") or not self._unquantized_part_sd:
            return

        for k, v in self._unquantized_part_sd.items():
            param = _get_nested_param(self, k)
            if param is None:
                continue
            param.data.copy_(v.to(device=param.device, dtype=param.dtype))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fuse_lowrank_pair(
    orig_packed_down: torch.Tensor,
    orig_packed_up: torch.Tensor,
    extra_down: torch.Tensor,
    extra_up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Unpack existing low-rank pair, concatenate with LoRA, repack.

    Returns ``(packed_down, packed_up, new_rank)``.
    """
    device = orig_packed_down.device
    dtype = orig_packed_down.dtype

    orig_down = unpack_lowrank_weight(orig_packed_down.data, down=True)
    orig_up = unpack_lowrank_weight(orig_packed_up.data, down=False)

    extra_down = extra_down.to(device=device, dtype=dtype)
    extra_up = extra_up.to(device=device, dtype=dtype)

    fused_down = torch.cat([orig_down, extra_down], dim=0)
    fused_up = torch.cat([orig_up, extra_up], dim=1)

    packed_down = pack_lowrank_weight(fused_down, down=True).contiguous()
    packed_up = pack_lowrank_weight(fused_up, down=False).contiguous()

    return packed_down, packed_up, fused_down.shape[0]


def _get_nested_attr(obj: Any, path: str) -> Any | None:
    """Safely traverse a dotted attribute path, returning *None* on failure."""
    for part in path.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (IndexError, TypeError, KeyError):
                return None
        else:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
    return obj


def _get_nested_param(model: nn.Module, key: str) -> nn.Parameter | None:
    """Resolve a state-dict key like ``'layers.0.adaLN_modulation.0.weight'`` to a Parameter."""
    parts = key.rsplit(".", 1)
    if len(parts) != 2:
        return None
    module = _get_nested_attr(model, parts[0])
    if module is None:
        return None
    return getattr(module, parts[1], None)


def _normalize_lora_format(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize various LoRA key formats to diffusers convention.

    Handles:
    - ``diffusion_model.`` prefix (ComfyUI)
    - ``transformer.`` prefix (diffusers pipeline-level keys)
    - ``lora_down``/``lora_up`` → ``lora_A``/``lora_B`` (kohya/ComfyUI)
    - ``.alpha`` scaling (kohya format)
    """
    sample_keys = list(sd.keys())[:5]
    logger.debug("LoRA normalization: %d input keys, samples: %s", len(sd), sample_keys)

    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("diffusion_model."):
            new_k = new_k[len("diffusion_model."):]
        if new_k.startswith("transformer."):
            new_k = new_k[len("transformer."):]
        out[new_k] = v
    sd = out

    has_kohya = any(".lora_down." in k or ".lora_up." in k for k in sd)
    has_diffusers = any(".lora_A." in k or ".lora_B." in k for k in sd)
    n_alphas = sum(1 for k in sd if k.endswith(".alpha"))
    logger.debug("LoRA format detect: kohya=%s, diffusers=%s, alphas=%d", has_kohya, has_diffusers, n_alphas)

    alpha_map: dict[str, torch.Tensor] = {}
    non_alpha: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.endswith(".alpha"):
            alpha_map[k] = v
        else:
            non_alpha[k] = v

    def _alpha_scale(down_weight: torch.Tensor, alpha_key: str) -> tuple[float, float]:
        alpha_tensor = alpha_map.get(alpha_key)
        if alpha_tensor is None:
            return 1.0, 1.0
        rank = down_weight.shape[0]
        scale = alpha_tensor.item() / rank
        # Distribute alpha/rank across down and up to keep magnitudes balanced.
        # Uses iterative halving instead of sqrt to stay in power-of-2 space,
        # which is friendlier to half-precision arithmetic.
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        return scale_down, scale_up

    if has_kohya:
        out: dict[str, torch.Tensor] = {}
        processed: set[str] = set()
        for k in list(non_alpha.keys()):
            if k in processed:
                continue
            if ".lora_down." in k:
                up_k = k.replace(".lora_down.", ".lora_up.")
                base = k.split(".lora_down.")[0]

                down_w = non_alpha[k]
                up_w = non_alpha.get(up_k)
                if up_w is None:
                    continue

                sd_down, sd_up = _alpha_scale(down_w, f"{base}.alpha")
                a_key = k.replace(".lora_down.", ".lora_A.")
                b_key = up_k.replace(".lora_up.", ".lora_B.")
                out[a_key] = down_w * sd_down
                out[b_key] = up_w * sd_up
                processed.update([k, up_k])
            elif ".lora_up." not in k:
                out[k] = non_alpha[k]
                processed.add(k)
        return out

    if has_diffusers and n_alphas > 0:
        out = {}
        processed: set[str] = set()
        for k in list(non_alpha.keys()):
            if k in processed:
                continue
            if ".lora_A." in k:
                b_k = k.replace(".lora_A.", ".lora_B.")
                base = k.split(".lora_A.")[0]

                down_w = non_alpha[k]
                up_w = non_alpha.get(b_k)
                if up_w is None:
                    out[k] = down_w
                    processed.add(k)
                    continue

                sd_down, sd_up = _alpha_scale(down_w, f"{base}.alpha")
                out[k] = down_w * sd_down
                out[b_k] = up_w * sd_up
                processed.update([k, b_k])
            elif ".lora_B." not in k:
                out[k] = non_alpha[k]
                processed.add(k)
        return out

    return {k: v for k, v in sd.items() if not k.endswith(".alpha")}
