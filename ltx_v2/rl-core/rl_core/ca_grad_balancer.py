from __future__ import annotations

import logging
import re

import torch

from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

logger = logging.getLogger(__name__)


def _scale_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return x
    return x.detach() + scale * (x - x.detach())


def parse_scale_config(value, default: float) -> tuple[float, dict[int, float]]:
    """Parse a scale config value that is either a scalar or a list of per-block rules.

    Returns ``(global_scale, {block_idx: scale})``.

    When *value* is a number, it becomes the global scale and there are no
    per-block overrides.  When *value* is a list, *default* is kept as the
    global scale and each entry produces per-block overrides::

        [
            {"blocks": [0],       "scale": 0.0},
            {"blocks": ["1-10"],  "scale": 0.5},
            {"blocks": ["40-47"], "scale": 0.3},
        ]
    """
    if value is None:
        return default, {}
    if isinstance(value, (int, float)):
        return float(value), {}
    if isinstance(value, (list, tuple)):
        overrides: dict[int, float] = {}
        for entry in value:
            scale = float(entry["scale"])
            for blk_spec in entry["blocks"]:
                s = str(blk_spec)
                if "-" in s:
                    lo, hi = s.split("-", 1)
                    for idx in range(int(lo), int(hi) + 1):
                        overrides[idx] = scale
                else:
                    overrides[int(s)] = scale
        return default, overrides
    return float(value), {}


class CrossAttentionGradBalancer:
    """Straight-through gradient scaler for cross-attention Q and KV inputs.

    Supports per-block per-component overrides via ``block_overrides``:
    a dict mapping ``(block_idx, direction, component)`` to a scale value,
    where *direction* is ``"a2v"`` or ``"v2a"`` and *component* is ``"q"``
    or ``"kv"``.  Overrides do NOT participate in warmup.

    Global scales support optional linear warmup via ``set_progress(t)``.
    """

    def __init__(
        self,
        q_scale_a2v: float = 1.0,
        kv_scale_a2v: float = 1.0,
        q_scale_v2a: float = 1.0,
        kv_scale_v2a: float = 1.0,
        warmup_frac: float = 0.0,
        q_scale_a2v_init: float | None = None,
        kv_scale_a2v_init: float | None = None,
        q_scale_v2a_init: float | None = None,
        kv_scale_v2a_init: float | None = None,
        block_overrides: dict[tuple[int, str, str], float] | None = None,
    ):
        self.hooks: list[torch.utils.hooks.RemovableHook] = []
        self.enabled = False
        self.warmup_frac = warmup_frac
        self._progress = 0.0
        self._block_overrides = block_overrides or {}

        self._scales_final = {
            "a2v_q": q_scale_a2v,
            "a2v_kv": kv_scale_a2v,
            "v2a_q": q_scale_v2a,
            "v2a_kv": kv_scale_v2a,
        }
        self._scales_init = {
            "a2v_q": q_scale_a2v_init if q_scale_a2v_init is not None else q_scale_a2v,
            "a2v_kv": kv_scale_a2v_init if kv_scale_a2v_init is not None else kv_scale_a2v,
            "v2a_q": q_scale_v2a_init if q_scale_v2a_init is not None else q_scale_v2a,
            "v2a_kv": kv_scale_v2a_init if kv_scale_v2a_init is not None else kv_scale_v2a,
        }

    def _current_scale(self, key: str) -> float:
        s0 = self._scales_init[key]
        s1 = self._scales_final[key]
        if self.warmup_frac <= 0.0 or s0 == s1:
            return s1
        t = min(self._progress / self.warmup_frac, 1.0)
        return s0 + (s1 - s0) * t

    def _layer_scale(self, block_idx: int, direction: str) -> tuple[float, float]:
        """Return (q_scale, kv_scale) for a specific block.

        Per-block overrides take precedence over the global (possibly
        warmed-up) value for the corresponding component.
        """
        q_scale = self._block_overrides.get(
            (block_idx, direction, "q"),
            self._current_scale(f"{direction}_q"),
        )
        kv_scale = self._block_overrides.get(
            (block_idx, direction, "kv"),
            self._current_scale(f"{direction}_kv"),
        )
        return q_scale, kv_scale

    def set_progress(self, progress: float):
        self._progress = max(0.0, min(1.0, progress))

    def register(self, model):
        block_pat = re.compile(r"\.(\d+)\.")
        max_block_idx = -1
        for name, module in model.named_modules():
            if isinstance(module, BasicAVTransformerBlock):
                m = block_pat.search(name)
                block_idx = int(m.group(1)) if m else -1
                max_block_idx = max(max_block_idx, block_idx)
                for attn_name, direction in (
                    ("audio_to_video_attn", "a2v"),
                    ("video_to_audio_attn", "v2a"),
                ):
                    attn = getattr(module, attn_name, None)
                    if attn is not None:
                        h = attn.register_forward_pre_hook(
                            self._make_hook(direction, block_idx),
                            with_kwargs=True,
                        )
                        self.hooks.append(h)

        override_blocks = sorted(set(idx for idx, _, _ in self._block_overrides))
        bad = [i for i in override_blocks if i > max_block_idx]
        if bad:
            logger.warning(
                f"CrossAttentionGradBalancer: override block indices {bad} "
                f"exceed max block index {max_block_idx}!"
            )

        log_parts = [
            f"registered {len(self.hooks)} hooks on {max_block_idx + 1} blocks",
            f"global: a2v(q={self._scales_final['a2v_q']}, kv={self._scales_final['a2v_kv']}), "
            f"v2a(q={self._scales_final['v2a_q']}, kv={self._scales_final['v2a_kv']})",
            f"warmup_frac={self.warmup_frac}",
        ]
        if self._block_overrides:
            log_parts.append(f"block_overrides({len(self._block_overrides)} entries)")
        logger.info(f"CrossAttentionGradBalancer: {', '.join(log_parts)}")

    def _make_hook(self, direction: str, block_idx: int):
        owner = self

        def hook_fn(module, args, kwargs):
            if not owner.enabled:
                return
            q_scale, kv_scale = owner._layer_scale(block_idx, direction)
            if q_scale == 1.0 and kv_scale == 1.0:
                return

            x = args[0] if len(args) > 0 else None
            ctx = kwargs.get("context", None)

            modified = False
            new_args = list(args)
            kwargs = dict(kwargs)

            if x is not None and x.requires_grad and q_scale != 1.0:
                new_args[0] = _scale_grad(x, q_scale)
                modified = True

            if ctx is not None and ctx.requires_grad and kv_scale != 1.0:
                kwargs["context"] = _scale_grad(ctx, kv_scale)
                modified = True

            if modified:
                return tuple(new_args), kwargs

        return hook_fn

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def build_ca_balancer_from_config(config, _cfg_get, model) -> CrossAttentionGradBalancer | None:
    """Build a CrossAttentionGradBalancer from config, or return None if all scales are default.

    Each of the four scale keys can be either a float (global) or a list of
    per-block rules::

        config.train.ca_kv_scale_a2v = [
            {"blocks": [0],       "scale": 0.0},
            {"blocks": ["1-10"],  "scale": 0.5},
            {"blocks": ["40-47"], "scale": 0.3},
        ]
    """
    ca_default = float(_cfg_get(config, "train.ca_grad_scale", 1.0))

    raw_q_a2v = _cfg_get(config, "train.ca_q_scale_a2v", _cfg_get(config, "train.ca_grad_scale_a2v", ca_default))
    raw_kv_a2v = _cfg_get(config, "train.ca_kv_scale_a2v", _cfg_get(config, "train.ca_grad_scale_a2v", ca_default))
    raw_q_v2a = _cfg_get(config, "train.ca_q_scale_v2a", _cfg_get(config, "train.ca_grad_scale_v2a", ca_default))
    raw_kv_v2a = _cfg_get(config, "train.ca_kv_scale_v2a", _cfg_get(config, "train.ca_grad_scale_v2a", ca_default))

    ca_q_a2v, q_a2v_blk = parse_scale_config(raw_q_a2v, ca_default)
    ca_kv_a2v, kv_a2v_blk = parse_scale_config(raw_kv_a2v, ca_default)
    ca_q_v2a, q_v2a_blk = parse_scale_config(raw_q_v2a, ca_default)
    ca_kv_v2a, kv_v2a_blk = parse_scale_config(raw_kv_v2a, ca_default)

    block_overrides: dict[tuple[int, str, str], float] = {}
    for idx, s in q_a2v_blk.items():
        block_overrides[(idx, "a2v", "q")] = s
    for idx, s in kv_a2v_blk.items():
        block_overrides[(idx, "a2v", "kv")] = s
    for idx, s in q_v2a_blk.items():
        block_overrides[(idx, "v2a", "q")] = s
    for idx, s in kv_v2a_blk.items():
        block_overrides[(idx, "v2a", "kv")] = s

    warmup = float(_cfg_get(config, "train.ca_warmup_frac", 0.0))
    q_a2v_init = _cfg_get(config, "train.ca_q_scale_a2v_init", None)
    kv_a2v_init = _cfg_get(config, "train.ca_kv_scale_a2v_init", None)
    q_v2a_init = _cfg_get(config, "train.ca_q_scale_v2a_init", None)
    kv_v2a_init = _cfg_get(config, "train.ca_kv_scale_v2a_init", None)

    any_non_default = any(s != 1.0 for s in [ca_q_a2v, ca_kv_a2v, ca_q_v2a, ca_kv_v2a])
    any_init = any(x is not None for x in [q_a2v_init, kv_a2v_init, q_v2a_init, kv_v2a_init])
    if not (any_non_default or any_init or block_overrides):
        return None

    balancer = CrossAttentionGradBalancer(
        q_scale_a2v=ca_q_a2v,
        kv_scale_a2v=ca_kv_a2v,
        q_scale_v2a=ca_q_v2a,
        kv_scale_v2a=ca_kv_v2a,
        warmup_frac=warmup,
        q_scale_a2v_init=float(q_a2v_init) if q_a2v_init is not None else None,
        kv_scale_a2v_init=float(kv_a2v_init) if kv_a2v_init is not None else None,
        q_scale_v2a_init=float(q_v2a_init) if q_v2a_init is not None else None,
        kv_scale_v2a_init=float(kv_v2a_init) if kv_v2a_init is not None else None,
        block_overrides=block_overrides,
    )
    balancer.register(model)
    return balancer
