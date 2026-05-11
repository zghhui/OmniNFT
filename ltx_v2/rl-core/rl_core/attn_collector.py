from __future__ import annotations

import logging
import math

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

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, BasicAVTransformerBlock):
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

    def get_weights(self, w_max: float, device: torch.device, num_frames: int = 0):
        if not self.layer_maps or w_max <= 1.0:
            return None, None

        block_avgs = []
        for block_idx in sorted(self.layer_maps.keys()):
            maps_list = self.layer_maps[block_idx]
            block_avgs.append(torch.stack(maps_list, dim=0).mean(dim=0))
        stacked = torch.stack(block_avgs, dim=0)
        avg = stacked.mean(dim=0)
        attn = avg[:1]

        video_scores = attn.sum(dim=1)
        audio_scores = attn.max(dim=2).values

        scale = w_max - 1.0

        def _soft_normalize(scores):
            mean = scores.mean(dim=-1, keepdim=True)
            std = scores.std(dim=-1, keepdim=True).clamp(min=1e-8)
            z = (scores - mean) / std
            normed = torch.sigmoid(z * scale)
            return (1.0 + scale * normed).clamp(min=1.0, max=w_max).unsqueeze(-1)

        if num_frames > 1:
            num_video_tokens = video_scores.shape[-1]
            spatial = num_video_tokens // num_frames
            framed = video_scores.view(1, num_frames, spatial)
            mean = framed.mean(dim=-1, keepdim=True)
            std = framed.std(dim=-1, keepdim=True).clamp(min=1e-8)
            z = (framed - mean) / std
            normed = torch.sigmoid(z * scale)
            v_w = (1.0 + scale * normed).clamp(min=1.0, max=w_max).view(1, num_video_tokens, 1).to(device)
        else:
            v_w = _soft_normalize(video_scores).to(device)

        a_w = _soft_normalize(audio_scores).to(device)
        return v_w, a_w
