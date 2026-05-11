from __future__ import annotations

import logging
import math

import numpy as np
import torch

from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

logger = logging.getLogger(__name__)


class V2AAttentionCollector:
    """Collect V2A cross-attention weights from denoising steps in [50%, 90%] range."""

    def __init__(self):
        self.hooks = []
        self.layer_maps: dict[int, list[torch.Tensor]] = {}
        self.enabled = False

    def _compute_attn_weights(self, module: Attention, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        b, seq_q, _ = q.shape
        heads = module.heads
        dim_head = module.dim_head
        q = q.view(b, seq_q, heads, dim_head).transpose(1, 2)
        k = k.view(b, -1, heads, dim_head).transpose(1, 2)
        scale = 1.0 / math.sqrt(dim_head)
        attn_probs = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        return attn_probs.mean(dim=1)

    def _make_hook(self, block_idx: int):
        collector = self

        def hook_fn(module: Attention, args, kwargs, output):
            if not collector.enabled:
                return
            x = args[0]
            context = kwargs.get("context", None)
            if context is None:
                context = x
            if kwargs.get("all_perturbed", False):
                return
            try:
                with torch.no_grad():
                    q = module.q_norm(module.to_q(x))
                    k = module.k_norm(module.to_k(context))
                    attn_weights = collector._compute_attn_weights(module, q, k)
                    if block_idx not in collector.layer_maps:
                        collector.layer_maps[block_idx] = []
                    collector.layer_maps[block_idx].append(attn_weights.cpu().float())
            except Exception as e:
                logger.warning(f"V2A hook error at block {block_idx}: {e}")

        return hook_fn

    def register_hooks(self, model, skip_first_n_layers: int = 10):
        for name, module in model.named_modules():
            if isinstance(module, BasicAVTransformerBlock):
                if module.idx < skip_first_n_layers:
                    continue
                if hasattr(module, "video_to_audio_attn"):
                    h = module.video_to_audio_attn.register_forward_hook(
                        self._make_hook(module.idx), with_kwargs=True,
                    )
                    self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def reset(self):
        self.layer_maps.clear()

    def get_weights(self, w_max: float, device: torch.device, num_frames: int = 0,
                    top_k_percent: float = 40.0):
        """Generate per-token weights using mask-based approach.

        1. Aggregate attention across layers and steps
        2. Compute spatial heatmap (clip top 1% outliers, normalize to [0,1])
        3. Binarize via top_k_percent threshold
        4. Mask region → w_max, non-mask region → 1.0
        """
        if not self.layer_maps or w_max <= 1.0:
            return None, None

        block_avgs = []
        for block_idx in sorted(self.layer_maps.keys()):
            maps_list = self.layer_maps[block_idx]
            block_avgs.append(torch.stack(maps_list, dim=0).mean(dim=0))
        stacked = torch.stack(block_avgs, dim=0)
        avg = stacked.mean(dim=0)
        attn = avg[:1]

        # --- Video weights (V2A: Q=audio, K=video → sum over audio queries per video token) ---
        video_scores = attn[:, :, :].sum(dim=1)  # (1, num_video_tokens)

        # --- Audio weights (V2A: max over video keys per audio token) ---
        audio_scores = attn.max(dim=2).values  # (1, num_audio_tokens)

        def _mask_weight(scores, num_spatial_frames=None, spatial_size=None):
            s = scores.squeeze(0).numpy()

            if num_spatial_frames is not None and spatial_size is not None:
                heatmap = s.reshape(num_spatial_frames, spatial_size)
            else:
                heatmap = s.reshape(1, -1)

            vmin = heatmap.min()
            vmax_clipped = float(np.percentile(heatmap, 99))
            if vmax_clipped > vmin:
                heatmap = np.clip(heatmap, vmin, vmax_clipped)
                heatmap = (heatmap - vmin) / (vmax_clipped - vmin)
            else:
                heatmap = np.zeros_like(heatmap)

            thresh_val = float(np.percentile(heatmap, 100.0 - top_k_percent))
            mask = (heatmap >= thresh_val).astype(float)

            weights = np.where(mask > 0, w_max, 1.0)
            return torch.from_numpy(weights.reshape(1, -1, 1)).float().to(device)

        if num_frames > 1:
            num_video_tokens = video_scores.shape[-1]
            spatial = num_video_tokens // num_frames
            v_w = _mask_weight(video_scores, num_spatial_frames=num_frames, spatial_size=spatial)
        else:
            v_w = _mask_weight(video_scores)

        a_w = _mask_weight(audio_scores)
        return v_w, a_w
