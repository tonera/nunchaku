"""
Compose multiple LoRA weights into a single LoRA dict.

This module operates entirely in **diffusers key space** (``lora_A.weight`` /
``lora_B.weight``), so it is model-agnostic — any model whose LoRAs follow the
standard diffusers naming convention can use :func:`compose_lora`.

Example
-------
>>> from nunchaku.lora.common.compose import compose_lora
>>> composed = compose_lora([
...     ("lora1.safetensors", 0.8),
...     ("lora2.safetensors", 1.0),
... ])
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ...utils import load_state_dict_in_safetensors


def compose_lora(
    loras: list[tuple[str | dict[str, torch.Tensor], float]],
    output_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """Compose multiple LoRA state dicts into one.

    Each input LoRA is scaled by its associated strength *before* being
    concatenated with others along the rank dimension.

    Parameters
    ----------
    loras : list of (path_or_dict, strength)
        Each element is a ``(source, strength)`` pair where *source* is either
        a path to a ``.safetensors`` file or a pre-loaded dict, and *strength*
        is the scaling factor.
    output_path : str, optional
        If given, the result is saved to this path.

    Returns
    -------
    dict[str, torch.Tensor]
        The composed LoRA weights (still in diffusers key space).
    """
    if len(loras) == 1:
        src, strength = loras[0]
        if isinstance(src, str):
            sd = load_state_dict_in_safetensors(src, device="cpu")
        else:
            sd = src
        sd = _normalize_compose_keyspace(sd)
        if abs(strength - 1.0) < 1e-6:
            return sd
        return _scale_lora(sd, strength)

    composed: dict[str, torch.Tensor] = {}
    for src, strength in loras:
        if isinstance(src, str):
            lora = load_state_dict_in_safetensors(src, device="cpu")
        else:
            lora = src

        lora = _normalize_compose_keyspace(lora)

        for k, v in lora.items():
            if v.ndim == 1:
                prev = composed.get(k)
                composed[k] = (v * strength) if prev is None else (prev + v * strength)
            elif v.ndim == 2:
                if _is_qkv_component(k, "q"):
                    if "lora_B" in k:
                        continue
                    _compose_qkv_group(composed, lora, k, strength)
                elif _is_qkv_component(k, "k") or _is_qkv_component(k, "v"):
                    continue
                else:
                    if "lora_A" in k:
                        v = v * strength
                    prev = composed.get(k)
                    if prev is None:
                        composed[k] = v
                    else:
                        dim = 0 if "lora_A" in k else 1
                        composed[k] = torch.cat([prev, v], dim=dim)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_file(composed, output_path)
    return composed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scale_lora(sd: dict[str, torch.Tensor], strength: float) -> dict[str, torch.Tensor]:
    """Scale ``lora_A`` and 1-D vectors by *strength*."""
    out = {}
    for k, v in sd.items():
        if v.ndim == 1 or "lora_A" in k:
            out[k] = v * strength
        else:
            out[k] = v
    return out


def _is_qkv_component(key: str, component: str) -> bool:
    """Check if *key* is a Q/K/V attention LoRA key."""
    return f".to_{component}." in key or f".add_{component}_proj." in key


def _qkv_sibling(key: str, src: str, dst: str) -> str:
    return key.replace(f".to_{src}.", f".to_{dst}.").replace(f".add_{src}_proj.", f".add_{dst}_proj.")


def _compose_qkv_group(
    composed: dict[str, torch.Tensor],
    lora: dict[str, torch.Tensor],
    q_a_key: str,
    strength: float,
) -> None:
    """Fuse Q/K/V lora_A and lora_B into a single ``to_qkv`` entry."""
    q_a = lora[q_a_key]
    k_a = lora.get(_qkv_sibling(q_a_key, "q", "k"))
    v_a = lora.get(_qkv_sibling(q_a_key, "q", "v"))
    if k_a is None or v_a is None:
        return

    q_b_key = q_a_key.replace("lora_A", "lora_B")
    q_b = lora[q_b_key]
    k_b = lora[_qkv_sibling(q_b_key, "q", "k")]
    v_b = lora[_qkv_sibling(q_b_key, "q", "v")]

    max_rank = max(q_a.shape[0], k_a.shape[0], v_a.shape[0])
    q_a = F.pad(q_a, (0, 0, 0, max_rank - q_a.shape[0]))
    k_a = F.pad(k_a, (0, 0, 0, max_rank - k_a.shape[0]))
    v_a = F.pad(v_a, (0, 0, 0, max_rank - v_a.shape[0]))
    q_b = F.pad(q_b, (0, max_rank - q_b.shape[1]))
    k_b = F.pad(k_b, (0, max_rank - k_b.shape[1]))
    v_b = F.pad(v_b, (0, max_rank - v_b.shape[1]))

    if torch.allclose(q_a, k_a) and torch.allclose(q_a, v_a):
        lora_a = q_a * strength
        lora_b = torch.cat([q_b, k_b, v_b], dim=0)
    else:
        all_a = [q_a, k_a, v_a]
        lora_a = torch.zeros(sum(a.shape[0] for a in all_a), q_a.shape[1], dtype=q_a.dtype)
        offset = 0
        for a in all_a:
            lora_a[offset : offset + a.shape[0]] = a
            offset += a.shape[0]
        lora_a = lora_a * strength

        all_b = [q_b, k_b, v_b]
        total_out = sum(b.shape[0] for b in all_b)
        total_rank = sum(b.shape[1] for b in all_b)
        lora_b = torch.zeros(total_out, total_rank, dtype=q_b.dtype)
        r_off, c_off = 0, 0
        for b in all_b:
            lora_b[r_off : r_off + b.shape[0], c_off : c_off + b.shape[1]] = b
            r_off += b.shape[0]
            c_off += b.shape[1]

    new_a_key = _qkv_sibling(q_a_key, "q", "qkv")
    new_b_key = new_a_key.replace("lora_A", "lora_B")

    for kk, vv, dim in ((new_a_key, lora_a, 0), (new_b_key, lora_b, 1)):
        prev = composed.get(kk)
        composed[kk] = vv if prev is None else torch.cat([prev, vv], dim=dim)


def _strip_diffusion_model_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("diffusion_model."):
            out[k[len("diffusion_model."):]] = v
        else:
            out[k] = v
    return out


def _strip_transformer_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("transformer."):
            out[k[len("transformer."):]] = v
        else:
            out[k] = v
    return out


def _strip_adapter_suffix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        new_k = k
        for marker in (".lora_A.", ".lora_B."):
            if marker not in k or not k.endswith(".weight"):
                continue
            prefix, suffix = k.split(marker, 1)
            if suffix == "weight":
                break
            adapter_name = suffix[: -len(".weight")]
            if not adapter_name:
                break
            new_k = f"{prefix}{marker[:-1]}.weight"
            break
        out[new_k] = v
    return out


def _normalize_compose_keyspace(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    sd = _strip_diffusion_model_prefix(sd)
    sd = _strip_transformer_prefix(sd)
    sd = _strip_adapter_suffix(sd)
    return sd
