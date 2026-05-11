"""GRPO/GDPO training for LTX audio-video model (v2.5 FSDP+DDP).

v2.5 changes over v2.4:
- Refactored V2AAttentionCollector and CrossAttentionGradBalancer into
  ltx_v2/rl-core/rl_core/ sub-modules to reduce main-script complexity.
- Per-block scale config: each CA scale key now accepts either a scalar
  float or a list of per-block rules, e.g.::

      config.train.ca_kv_scale_a2v = [
          {"blocks": [0],       "scale": 0.0},
          {"blocks": ["1-10"],  "scale": 0.5},
          {"blocks": ["40-47"], "scale": 0.3},
      ]

  Config keys (scalar or per-block list):
    train.ca_q_scale_a2v   (A2V Q=video side, default 1.0)
    train.ca_kv_scale_a2v  (A2V KV=audio side, default 1.0)
    train.ca_q_scale_v2a   (V2A Q=audio side, default 1.0)
    train.ca_kv_scale_v2a  (V2A KV=video side, default 1.0)
    train.ca_warmup_frac   (linear warmup from *_init to final, default 0.0)
    train.ca_*_init        (optional init values for warmup)

Migrated to ltx_v2 API: uses ltx_trainer.model_loader + ValidationSampler.
Algorithm: V2AAttentionCollector, branch_aware/gdpo,
attn_sync per-token weighting, LoRA dual adapter, EMA, TensorBoard.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure ltx_v2 packages are importable (ltx-core, ltx-pipelines, ltx-trainer)
# ---------------------------------------------------------------------------
_LTX_V2_ROOT = Path(__file__).resolve().parents[1] / "ltx_v2"
for _pkg in ("ltx-core", "ltx-pipelines", "ltx-trainer"):
    _src = str(_LTX_V2_ROOT / _pkg / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
_rl_core_root = str(_LTX_V2_ROOT / "rl-core")
if _rl_core_root not in sys.path:
    sys.path.insert(0, _rl_core_root)

import flow_grpo.rewards
import numpy as np
import torch
import torch.distributed as dist
import torchaudio
import tqdm
from absl import app, flags
from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.fsdp_utils import (
    FSDPConfig,
    fsdp_wrapper,
    register_optimizer_offload_hooks,
    save_fsdp_checkpoint,
)
from flow_grpo.stat_tracking import PerPromptStatTracker

# ltx-core components
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoLatentShape
from ltx_core.utils import to_denoised

# ltx-trainer components
from ltx_trainer.model_loader import load_embeddings_processor, load_model, load_transformer
from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler

# ltx-pipelines (needed for encode_video used by reward functions)
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
from ltx_pipelines.utils.media_io import encode_video

from rl_core import V2AAttentionCollector, build_ca_balancer_from_config

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
tqdm = tqdm.tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# =============================================================================
# Utility functions (self-contained)
# =============================================================================


def _cfg_get(config, key: str, default=None):
    cur = config
    for part in key.split("."):
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def setup_distributed(rank: int, local_rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def _safe_prompt_name(prompt: str, max_len: int = 15) -> str:
    name = "".join(c if c.isalnum() else "_" for c in str(prompt)[:max_len])
    return name if name else "empty_prompt"


def gather_tensor_to_all(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor.detach().cpu()
    outputs = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(outputs, tensor)
    return torch.cat(outputs, dim=0).detach().cpu()


def gather_list_to_all(items: list[str], world_size: int) -> list[str]:
    if world_size == 1:
        return list(items)
    outputs: list[list[str]] = [None for _ in range(world_size)]
    dist.all_gather_object(outputs, items)
    merged: list[str] = []
    for x in outputs:
        merged.extend(x)
    return merged


def return_decay(step: int, decay_type: int) -> float:
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"unsupported decay_type={decay_type}")
    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def _normalize_route_keys(raw_keys) -> set[str]:
    if raw_keys is None:
        return set()
    if isinstance(raw_keys, str):
        return {raw_keys.lower()}
    if isinstance(raw_keys, (list, tuple, set)):
        return {str(x).lower() for x in raw_keys}
    try:
        return {str(x).lower() for x in list(raw_keys)}
    except Exception:
        return set()


def _infer_reward_route(reward_name: str, video_keys: set[str], audio_keys: set[str], sync_keys: set[str]) -> str:
    name = reward_name.lower().strip()
    if name in sync_keys:
        return "sync"
    if name in audio_keys:
        return "audio"
    if name in video_keys:
        return "video"
    return "sync"


# =============================================================================
# Dataset
# =============================================================================


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        if not os.path.exists(self.file_path):
            self.file_path = dataset
        self.metadatas = []
        self.prompts_v = []
        self.prompts_a = []
        self.prompts_va = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                self.metadatas.append(item)
                # 三种 prompt（缺失时给空串，避免崩）
                self.prompts_v.append(item.get("prompt_v", ""))
                self.prompts_a.append(item.get("prompt_a", ""))
                self.prompts_va.append(item.get("prompt_av", ""))
    def __len__(self):
        return len(self.prompts_va)
    def __getitem__(self, idx):
        return {
            "prompt_v": self.prompts_v[idx],
            "prompt_a": self.prompts_a[idx],
            "prompt_va": self.prompts_va[idx],
            "metadata": self.metadatas[idx],
        }
    @staticmethod
    def collate_fn(examples):
        prompts_v = [e["prompt_v"] for e in examples]
        prompts_a = [e["prompt_a"] for e in examples]
        prompts_va = [e["prompt_va"] for e in examples]
        metadatas = [e["metadata"] for e in examples]
        return prompts_va, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, "k must divide num_replicas*batch_size"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            picked = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated = [idx for idx in picked for _ in range(self.k)]
            shuffle = torch.randperm(len(repeated), generator=g).tolist()
            repeated = [repeated[i] for i in shuffle]
            shards = []
            for i in range(self.num_replicas):
                st = i * self.batch_size
                ed = st + self.batch_size
                shards.append(repeated[st:ed])
            yield shards[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


# =============================================================================
# RuntimeArgs (simplified: scalar guidance instead of MultiModalGuiderParams)
# =============================================================================


@dataclass
class RuntimeArgs:
    checkpoint_path: str
    gemma_root: str
    height: int
    width: int
    num_frames: int
    frame_rate: float
    negative_prompt: str
    guidance_scale: float
    video_guidance_scale: float
    audio_guidance_scale: float
    stg_scale: float
    stg_blocks: list[int]
    stg_mode: str
    images: list[ImageConditioningInput]
    enhance_prompt: bool
    num_inference_steps: int


def _parse_images(raw_images) -> list[ImageConditioningInput]:
    if raw_images is None:
        return []
    out = []
    for item in raw_images:
        if isinstance(item, ImageConditioningInput):
            out.append(item)
        elif isinstance(item, dict):
            out.append(
                ImageConditioningInput(
                    path=item["path"],
                    frame_idx=int(item.get("frame_idx", 0)),
                    strength=float(item.get("strength", 1.0)),
                    crf=int(item.get("crf", 33)),
                )
            )
        elif isinstance(item, (tuple, list)) and len(item) >= 3:
            out.append(
                ImageConditioningInput(
                    path=str(item[0]),
                    frame_idx=int(item[1]),
                    strength=float(item[2]),
                    crf=int(item[3]) if len(item) > 3 else 33,
                )
            )
    return out


def build_runtime_args(config, *, for_eval: bool) -> RuntimeArgs:
    checkpoint_path = _cfg_get(config, "pretrained.model", "")
    gemma_root = _cfg_get(config, "gemma_root", "")
    if not checkpoint_path:
        raise ValueError("missing config.pretrained.model for LTX checkpoint")
    if not gemma_root:
        raise ValueError("missing config.gemma_root")

    params = detect_params(checkpoint_path)
    height = int(_cfg_get(config, "resolution_height", params.stage_1_height))
    width = int(_cfg_get(config, "resolution_width", params.stage_1_width))

    if for_eval:
        guidance_scale = float(
            _cfg_get(
                config,
                "sample.eval_video_cfg_guidance_scale",
                _cfg_get(config, "sample.video_cfg_guidance_scale", params.video_guider_params.cfg_scale),
            )
        )
        video_guidance_scale = float(
            _cfg_get(
                config,
                "sample.eval_video_cfg_guidance_scale",
                _cfg_get(config, "sample.video_cfg_guidance_scale", params.video_guider_params.cfg_scale),
            )
        )
        audio_guidance_scale = float(
            _cfg_get(
                config,
                "sample.eval_audio_cfg_guidance_scale",
                _cfg_get(config, "sample.audio_cfg_guidance_scale", params.audio_guider_params.cfg_scale),
            )
        )
        num_inference_steps = int(_cfg_get(config, "sample.eval_num_steps", params.num_inference_steps))
    else:
        guidance_scale = float(_cfg_get(config, "sample.train_video_cfg_guidance_scale", 1.0))
        # 默认采样cfg是1.5
        video_guidance_scale = float(_cfg_get(config, "sample.train_video_cfg_guidance_scale", 1.5))
        # 默认采样cfg是3
        audio_guidance_scale = float(_cfg_get(config, "sample.train_audio_cfg_guidance_scale", 3.0))
        num_inference_steps = int(_cfg_get(config, "sample.num_steps", params.num_inference_steps))

    stg_scale = float(_cfg_get(config, "sample.video_stg_scale", params.video_guider_params.stg_scale))
    stg_blocks = list(_cfg_get(config, "sample.video_stg_blocks", params.video_guider_params.stg_blocks))
    stg_mode = str(_cfg_get(config, "sample.stg_mode", "stg_av"))

    return RuntimeArgs(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        height=height,
        width=width,
        num_frames=int(_cfg_get(config, "sample.num_frames", params.num_frames)),
        frame_rate=float(_cfg_get(config, "sample.frame_rate", params.frame_rate)),
        negative_prompt=str(_cfg_get(config, "sample.negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
        guidance_scale=guidance_scale,
        video_guidance_scale=video_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        stg_mode=stg_mode,
        images=_parse_images(_cfg_get(config, "sample.images", [])),
        enhance_prompt=bool(_cfg_get(config, "sample.enhance_prompt", False)),
        num_inference_steps=num_inference_steps,
    )

# =============================================================================
# sample_one: rewritten using ValidationSampler internal
# =============================================================================

def sample_one(
    sampler: ValidationSampler,
    transformer,
    prompt: str,
    runtime: RuntimeArgs,
    seed: int,
    device: torch.device,
    attn_collector: V2AAttentionCollector | None = None,
):
    condition_image = None

    gen_config = GenerationConfig(
        prompt=prompt,
        negative_prompt=runtime.negative_prompt,
        height=runtime.height,
        width=runtime.width,
        num_frames=runtime.num_frames,
        frame_rate=runtime.frame_rate,
        num_inference_steps=runtime.num_inference_steps,
        guidance_scale=runtime.guidance_scale,
        video_guidance_scale=runtime.video_guidance_scale,
        audio_guidance_scale=runtime.audio_guidance_scale,
        seed=seed,
        condition_image=condition_image,
        generate_audio=True,
        stg_scale=runtime.stg_scale,
        stg_blocks=runtime.stg_blocks if runtime.stg_blocks else None,
        stg_mode=runtime.stg_mode,
    )

    sampler._validate_config(gen_config)
    v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = sampler._get_prompt_embeddings(gen_config, device)

    video_tools = sampler._create_video_latent_tools(gen_config)
    audio_tools = sampler._create_audio_latent_tools(gen_config)
    generator = torch.Generator(device=device).manual_seed(seed)

    video_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
    audio_clean_state = audio_tools.create_initial_state(device=device, dtype=torch.bfloat16)

    if gen_config.condition_image is not None:
        video_clean_state = sampler._apply_image_conditioning(
            video_clean_state, gen_config.condition_image, gen_config, device
        )

    noiser = GaussianNoiser(generator=generator)
    video_state = noiser(latent_state=video_clean_state, noise_scale=1.0)
    audio_state = noiser(latent_state=audio_clean_state, noise_scale=1.0)

    def _attn_last_step_cb(step_idx: int, total_steps: int) -> None:
        if attn_collector is not None:
            start = int(total_steps * 0.5)
            end = int(total_steps * 0.9)
            attn_collector.enabled = (start <= step_idx <= end)

    orig_transformer = sampler._transformer
    sampler._transformer = transformer
    try:
        video_state, audio_state = sampler._run_denoising(
            config=gen_config,
            video_state=video_state,
            audio_state=audio_state,
            video_clean_state=video_clean_state,
            audio_clean_state=audio_clean_state,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            device=device,
            step_callback=_attn_last_step_cb,
        )
    finally:
        sampler._transformer = orig_transformer

    video_unpatch_state = video_tools.clear_conditioning(video_state)
    video_unpatch_state = video_tools.unpatchify(video_unpatch_state)

    audio_unpatch_state = audio_tools.clear_conditioning(audio_state)
    audio_unpatch_state = audio_tools.unpatchify(audio_unpatch_state)

    video_float = sampler._decode_video(video_unpatch_state, device, gen_config.tiled_decoding)
    video_out = (video_float.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    video_out = video_out.permute(1, 2, 3, 0)

    audio_waveform = sampler._decode_audio(audio_unpatch_state, device)
    audio_sr = sampler._vocoder.output_sampling_rate if sampler._vocoder is not None else 16000
    audio_out = Audio(waveform=audio_waveform, sampling_rate=audio_sr) if audio_waveform is not None else None

    return {
        "video": video_out,
        "audio": audio_out,
        "video_latent": video_unpatch_state.latent.detach(),
        "audio_latent": audio_unpatch_state.latent.detach(),
        "v_context": v_ctx_pos.detach(),
        "a_context": a_ctx_pos.detach(),
    }


# =============================================================================
# Model helpers
# =============================================================================


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _set_adapter(model, adapter_name: str):
    base = _unwrap_model(model)
    if isinstance(base, PeftModel):
        base.set_adapter(adapter_name)


def _disable_adapter_ctx(model):
    base = _unwrap_model(model)
    if isinstance(base, PeftModel):
        return base.disable_adapter()
    return nullcontext()


def _collect_trainable_params(model):
    return [p for p in _unwrap_model(model).parameters() if p.requires_grad]


def _get_fsdp_auto_wrap_layer_cls():
    try:
        return {BasicAVTransformerBlock}
    except Exception as e:
        logger.warning(f"Cannot use BasicAVTransformerBlock for FSDP auto_wrap: {e}")
        return set()


# =============================================================================
# Checkpoint saving (unified FSDP + DDP)
# =============================================================================


def _save_lora_from_fsdp(model_engine: FSDP, save_dir: Path, rank: int, adapter_name: str = "default"):
    save_dir.mkdir(parents=True, exist_ok=True)
    state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model_engine, StateDictType.FULL_STATE_DICT, state_cfg):
        full_state = model_engine.state_dict()
    if rank == 0:
        model_to_save = _unwrap_model(model_engine)
        if not isinstance(model_to_save, PeftModel):
            raise RuntimeError("Model is not PeftModel, cannot save LoRA from FSDP.")
        try:
            model_to_save.set_adapter(adapter_name)
        except Exception:
            pass
        lora_state = get_peft_model_state_dict(model_to_save, state_dict=full_state, adapter_name=adapter_name)
        if len(lora_state) == 0:
            raise RuntimeError(f"LoRA state_dict is empty for adapter '{adapter_name}'.")
        save_file(lora_state, str(save_dir / "adapter_model.safetensors"))
        peft_cfg = model_to_save.peft_config.get(adapter_name, None)
        if peft_cfg is None:
            raise RuntimeError(f"adapter config not found for '{adapter_name}'.")
        peft_cfg.save_pretrained(str(save_dir))


def save_ckpt(
    log_dir,
    model_engine,
    optimizer,
    scaler,
    global_step,
    rank,
    trainable_params,
    ema,
    use_fsdp: bool = False,
    use_lora: bool = True,
    epoch: int = 0,
):
    if use_fsdp:
        ckpt_dir = Path(f"{log_dir}/checkpoints/checkpoint-{global_step}")
        if rank == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if ema is not None:
                torch.save(ema.state_dict(), ckpt_dir / "ema_state.pt")
        dist.barrier()

        if ema is not None:
            ema.copy_ema_to(trainable_params, store_temp=True)

        if use_lora:
            _save_lora_from_fsdp(model_engine, ckpt_dir / "lora", rank, adapter_name="default")
            _save_lora_from_fsdp(model_engine, ckpt_dir / "lora_old", rank, adapter_name="old")
        else:
            save_fsdp_checkpoint(str(Path(log_dir) / "checkpoints"), model_engine, global_step, rank)

        if use_lora:
            optim_sd = optimizer.state_dict()
        else:
            optim_sd = FSDP.optim_state_dict(model_engine, optimizer)
        if rank == 0:
            state = {
                "optimizer": optim_sd,
                "global_step": global_step,
                "use_lora": use_lora,
                "epoch": epoch,
            }
            if scaler is not None and scaler.is_enabled():
                state["scaler"] = scaler.state_dict()
            torch.save(state, ckpt_dir / "train_state.pt")
            logger.info(f"Saved FSDP checkpoint to {ckpt_dir}")
        dist.barrier()

        if ema is not None:
            ema.copy_temp_to(trainable_params)
    else:
        if ema is not None:
            ema.copy_ema_to(trainable_params, store_temp=True)

        if is_main_process(rank):
            ckpt_dir = Path(f"{log_dir}/checkpoints/checkpoint-{global_step}")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_to_save = _unwrap_model(model_engine)
            lora_dir = ckpt_dir / "lora"
            lora_dir.mkdir(parents=True, exist_ok=True)
            model_to_save.save_pretrained(lora_dir)
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            if scaler is not None:
                torch.save(scaler.state_dict(), ckpt_dir / "scaler.pt")
            logger.info(f"Saved DDP checkpoint to {ckpt_dir}")

        if ema is not None:
            ema.copy_temp_to(trainable_params)


# =============================================================================
# Main training loop
# =============================================================================


def main(_):
    config = FLAGS.config
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(int(_cfg_get(config, "seed", 42)), rank)

    if bool(_cfg_get(config, "allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    runtime_train = build_runtime_args(config, for_eval=False)
    runtime_eval = build_runtime_args(config, for_eval=True)

    base_name = _cfg_get(config, "run_name", "nft_ltx")
    config.run_name = f"{base_name}"

    writer = None
    log_dir = os.path.join(_cfg_get(config, "logdir", "logs"), config.run_name)
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    if is_main_process(rank):
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_text("config", str(config.to_dict()), 0)

    # =========================================================================
    # Mixed precision & GradScaler
    # =========================================================================
    mixed = _cfg_get(config, "mixed_precision", "bf16")
    mp_dtype = torch.float16 if mixed == "fp16" else (torch.bfloat16 if mixed == "bf16" else None)
    enable_amp = mp_dtype is not None
    use_fp16_scaler = mp_dtype == torch.float16
    scaler = GradScaler(enabled=use_fp16_scaler)

    # =========================================================================
    # Model loading via ltx-trainer (replaces TI2VidOneStagePipeline)
    # =========================================================================
    _load_dtype = mp_dtype if mp_dtype is not None else torch.bfloat16
    components = load_model(
        checkpoint_path=runtime_train.checkpoint_path,
        text_encoder_path=runtime_train.gemma_root,
        device="cpu",
        dtype=_load_dtype,
        with_video_vae_encoder=True,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=True,
        with_vocoder=True,
        with_text_encoder=True,
    )
    embeddings_processor = load_embeddings_processor(
        checkpoint_path=runtime_train.checkpoint_path,
        device="cpu",
        dtype=_load_dtype,
    )

    video_patchifier = VideoLatentPatchifier(patch_size=1)
    audio_patchifier = AudioPatchifier(patch_size=1)

    use_lora = bool(_cfg_get(config, "use_lora", True))
    use_fsdp = bool(_cfg_get(config, "train.use_fsdp", True))

    # =========================================================================
    # Resume state
    # =========================================================================
    resume_from = _cfg_get(config, "resume_from", "")
    resume_root = Path(resume_from) if resume_from else None
    resume_state_path = (resume_root / "train_state.pt") if resume_root is not None else None
    resume_lora_path = (resume_root / "lora") if resume_root is not None else None
    resume_lora_old_path = (resume_root / "lora_old") if resume_root is not None else None
    resume_ema_path = (resume_root / "ema_state.pt") if resume_root is not None else None
    resume_model_path = (resume_root / "model.safetensors") if resume_root is not None else None
    old_adapter_restored = False

    # =========================================================================
    # Transformer setup
    # =========================================================================
    model = components.transformer.to("cpu", dtype=torch.float32).train()
    old_model = None
    ref_model = None
    old_trainable_params = None

    global_step = 0
    start_epoch = 0
    resume_optimizer_state = None
    resume_scaler_state = None
    if resume_state_path is not None and resume_state_path.exists():
        state = torch.load(resume_state_path, map_location="cpu")
        resume_optimizer_state = state.get("optimizer", None)
        resume_scaler_state = state.get("scaler", None)
        global_step = int(state.get("global_step", 0))
        start_epoch = int(state.get("epoch", 0)) + 1
        if (not use_lora) and state.get("model", None) is not None:
            model.load_state_dict(state["model"], strict=False)
        logger.info(f"resume from {resume_state_path}, global_step={global_step}, start_epoch={start_epoch}")

    if (not use_lora) and resume_model_path is not None and resume_model_path.exists():
        logger.info(f"loading non-LoRA model from {resume_model_path}")
        model_state = load_file(str(resume_model_path))
        model.load_state_dict(model_state, strict=False)

    if use_lora:
        target_modules = list(_cfg_get(config, "target_modules", ["to_q", "to_k", "to_v", "to_out.0"]))
        lora_cfg = LoraConfig(
            r=int(_cfg_get(config, "train.lora_r", 32)),
            lora_alpha=int(_cfg_get(config, "train.lora_alpha", 64)),
            init_lora_weights=str(_cfg_get(config, "train.lora_init", "gaussian")),
            target_modules=target_modules,
        )
        init_lora_path = _cfg_get(config, "train.lora_path", None)
        if resume_lora_path is not None and resume_lora_path.exists():
            init_lora_path = str(resume_lora_path)

        if init_lora_path:
            model = PeftModel.from_pretrained(model, init_lora_path, is_trainable=True)
            model.set_adapter("default")
        else:
            model = get_peft_model(model, lora_cfg)

        adapter_names = set(model.peft_config.keys())
        if "old" not in adapter_names:
            model.add_adapter("old", lora_cfg)
        if resume_lora_old_path is not None and resume_lora_old_path.exists():
            old_lora_state = load_file(str(resume_lora_old_path / "adapter_model.safetensors"))
            set_peft_model_state_dict(model, old_lora_state, adapter_name="old")
            old_adapter_restored = True
            logger.info(f"Restored 'old' adapter from {resume_lora_old_path}")
        model.set_adapter("default")
    else:
        logger.warning("use_lora=False: high VRAM usage, may OOM.")
        model.requires_grad_(True)
        old_model = load_transformer(
            runtime_train.checkpoint_path, "cpu", mp_dtype if mp_dtype is not None else torch.float32
        ).eval().requires_grad_(False)
        ref_model = load_transformer(
            runtime_train.checkpoint_path, "cpu", mp_dtype if mp_dtype is not None else torch.float32
        ).eval().requires_grad_(False)

    # =========================================================================
    # Create ValidationSampler
    # =========================================================================
    sampler = ValidationSampler(
        transformer=model,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=components.video_vae_encoder,
        text_encoder=components.text_encoder,
        embeddings_processor=embeddings_processor,
        audio_decoder=components.audio_vae_decoder,
        vocoder=components.vocoder,
    )

    # =========================================================================
    # FSDP / DDP wrapping
    # =========================================================================
    if use_fsdp:
        fsdp_config = FSDPConfig(
            sharding_strategy=str(_cfg_get(config, "train.fsdp_sharding_strategy", "SHARD_GRAD_OP")).upper(),
            backward_prefetch=str(_cfg_get(config, "train.fsdp_backward_prefetch", "BACKWARD_PRE")).upper(),
            cpu_offload=bool(_cfg_get(config, "train.fsdp_cpu_offload", False)),
            num_replicate=int(_cfg_get(config, "train.fsdp_num_replicate", 1)),
            num_shard=int(_cfg_get(config, "train.fsdp_num_shard", world_size)),
            mixed_precision_dtype=mp_dtype,
            use_activation_checkpointing=bool(_cfg_get(config, "activation_checkpointing", True)),
            use_device_mesh=bool(_cfg_get(config, "train.fsdp_use_device_mesh", False)),
        )
        model.cpu().to(dtype=torch.float32)
        model_engine = fsdp_wrapper(model, fsdp_config, _get_fsdp_auto_wrap_layer_cls)

        if not use_lora:
            eval_fsdp_config = FSDPConfig(
                sharding_strategy=str(_cfg_get(config, "train.fsdp_sharding_strategy", "SHARD_GRAD_OP")).upper(),
                backward_prefetch=str(_cfg_get(config, "train.fsdp_backward_prefetch", "BACKWARD_PRE")).upper(),
                cpu_offload=bool(_cfg_get(config, "train.fsdp_cpu_offload", False)),
                num_replicate=int(_cfg_get(config, "train.fsdp_num_replicate", 1)),
                num_shard=int(_cfg_get(config, "train.fsdp_num_shard", world_size)),
                mixed_precision_dtype=mp_dtype,
                use_activation_checkpointing=False,
                use_device_mesh=bool(_cfg_get(config, "train.fsdp_use_device_mesh", False)),
            )
            old_model.cpu().to(dtype=torch.float32)
            ref_model.cpu().to(dtype=torch.float32)
            old_model = fsdp_wrapper(old_model, eval_fsdp_config, _get_fsdp_auto_wrap_layer_cls)
            ref_model = fsdp_wrapper(ref_model, eval_fsdp_config, _get_fsdp_auto_wrap_layer_cls)
    else:
        _ddp_dtype = mp_dtype if mp_dtype is not None else torch.float32
        model_engine = DDP(
            model.to(device, dtype=_ddp_dtype),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(_cfg_get(config, "train.ddp_find_unused_parameters", True)),
        )
        if not use_lora:
            old_model = old_model.to(device, dtype=_ddp_dtype)
            ref_model = ref_model.to(device, dtype=_ddp_dtype)

    # =========================================================================
    # Trainable params & optimizer
    # =========================================================================
    if use_lora:
        _set_adapter(model_engine, "default")
        if use_fsdp:
            fsdp_params_dict = dict(model_engine.named_parameters())
            trainable_params = []
            old_trainable_params = []
            for name, param in fsdp_params_dict.items():
                if "default" in name and param.requires_grad:
                    old_name = name.replace("default", "old")
                    if old_name in fsdp_params_dict:
                        trainable_params.append(param)
                        old_trainable_params.append(fsdp_params_dict[old_name])
                    else:
                        raise ValueError(f"FSDP mapping error: {old_name} not found for {name}")
        else:
            trainable_params = _collect_trainable_params(model_engine)
            _set_adapter(model_engine, "old")
            old_trainable_params = _collect_trainable_params(model_engine)
            _set_adapter(model_engine, "default")
        if len(trainable_params) != len(old_trainable_params):
            raise RuntimeError("default/old adapter param count mismatch.")
        if is_main_process(rank):
            logger.info(
                f"LoRA trainable params (default): {len(trainable_params)}, "
                f"total numel: {sum(p.numel() for p in trainable_params)}"
            )
    else:
        trainable_params = _collect_trainable_params(model_engine)

    if len(trainable_params) == 0:
        raise RuntimeError("No trainable params. Check LoRA config or requires_grad.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(_cfg_get(config, "train.learning_rate", 3e-4)),
        betas=(float(_cfg_get(config, "train.adam_beta1", 0.9)), float(_cfg_get(config, "train.adam_beta2", 0.999))),
        weight_decay=float(_cfg_get(config, "train.adam_weight_decay", 1e-4)),
        eps=float(_cfg_get(config, "train.adam_epsilon", 1e-8)),
    )
    if resume_optimizer_state is not None:
        if use_fsdp and not use_lora:
            optim_sd_to_load = FSDP.optim_state_dict_to_load(model_engine, optimizer, resume_optimizer_state)
            optimizer.load_state_dict(optim_sd_to_load)
        else:
            optimizer.load_state_dict(resume_optimizer_state)
    if scaler.is_enabled() and resume_scaler_state is not None:
        scaler.load_state_dict(resume_scaler_state)

    if use_fsdp and bool(_cfg_get(config, "train.fsdp_optimizer_offload", False)):
        register_optimizer_offload_hooks(optimizer)
        logger.info("FSDP optimizer offload hooks enabled.")

    ema = (
        EMAModuleWrapper(trainable_params, decay=0.9, update_step_interval=1, device=device)
        if _cfg_get(config, "train.ema", True)
        else None
    )
    if ema is not None and resume_ema_path is not None and resume_ema_path.exists():
        ema.load_state_dict(torch.load(resume_ema_path, map_location="cpu"))
        logger.info(f"Restored EMA state from {resume_ema_path}")

    # =========================================================================
    # Data
    # =========================================================================
    train_dataset = TextPromptDataset(_cfg_get(config, "train_dataset", ""), "train")
    test_dataset = TextPromptDataset(_cfg_get(config, "test_dataset", ""), "test")

    train_sampler = DistributedKRepeatSampler(
        train_dataset,
        batch_size=int(_cfg_get(config, "sample.train_batch_size", 1)),
        k=int(_cfg_get(config, "sample.num_image_per_prompt", 1)),
        num_replicas=world_size,
        rank=rank,
        seed=int(_cfg_get(config, "seed", 42)),
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=train_dataset.collate_fn, num_workers=0)

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(_cfg_get(config, "sample.test_batch_size", 1)),
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
    )

    # =========================================================================
    # Training config
    # =========================================================================
    reward_fn = flow_grpo.rewards.multi_score(device, _cfg_get(config, "reward_fn", {}))
    use_video_reward = True
    stat_tracker = PerPromptStatTracker(bool(_cfg_get(config, "sample.global_std", True)))

    sigmas_all = LTX2Scheduler().execute(int(_cfg_get(config, "sample.num_steps", runtime_train.num_inference_steps))).to(
        device=device, dtype=torch.float32
    )
    if sigmas_all[-1].item() == 0:
        sigmas_all = sigmas_all[:-1]
    num_train_t = max(1, int(len(sigmas_all) * float(_cfg_get(config, "train.timestep_fraction", 1.0))))
    sigmas_train = sigmas_all[:num_train_t]

    grad_accum = int(_cfg_get(config, "train.gradient_accumulation_steps", 1))
    eff_accum = grad_accum * num_train_t
    train_bs = int(_cfg_get(config, "train.batch_size", 1))
    num_inner = int(_cfg_get(config, "train.num_inner_epochs", 1))
    num_batches_per_epoch = int(_cfg_get(config, "sample.num_batches_per_epoch", 1))
    num_epochs = int(_cfg_get(config, "num_epochs", 100000))
    save_freq = int(_cfg_get(config, "save_freq", 30))
    eval_freq = int(_cfg_get(config, "eval_freq", 10))
    debug = bool(_cfg_get(config, "debug", False))

    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    temp_dir = os.path.join(log_dir, "temp_video")
    os.makedirs(temp_dir, exist_ok=True)

    attn_sync_w_max = float(_cfg_get(config, "train.attn_sync_weight_max", 1.0))
    attn_sync_warmup_frac = float(_cfg_get(config, "train.attn_sync_warmup_frac", 0.5))
    attn_sync_warmup_steps = int(_cfg_get(config, "train.attn_sync_warmup_steps", 0))
    use_attn_sync = attn_sync_w_max > 1.0
    total_steps_for_schedule = max(num_epochs * num_batches_per_epoch, 1)

    ca_balancer = build_ca_balancer_from_config(config, _cfg_get, _unwrap_model(model_engine))

    if use_lora and not old_adapter_restored:
        if use_fsdp:
            with FSDP.summon_full_params(model_engine, writeback=True):
                with torch.no_grad():
                    fsdp_params = dict(model_engine.named_parameters())
                    for name, src_param in fsdp_params.items():
                        if "default" in name and src_param.requires_grad:
                            old_name = name.replace("default", "old")
                            if old_name in fsdp_params:
                                fsdp_params[old_name].copy_(src_param)
        else:
            for src_param, tgt_param in zip(trainable_params, old_trainable_params, strict=True):
                tgt_param.data.copy_(src_param.detach().data)

    # =========================================================================
    # Training epochs
    # =========================================================================
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        model_engine.eval()
        if ca_balancer is not None:
            ca_balancer.enabled = False
        if use_lora:
            _set_adapter(model_engine, "old")
        samples = []

        save_every = int(_cfg_get(config, "sample.save_every", 10))
        should_save = epoch % save_every == 0
        sample_save_dir = None
        if should_save:
            sample_save_dir = os.path.join(log_dir, "train_sampled_videos", f"epoch_{epoch}")
            os.makedirs(sample_save_dir, exist_ok=True)

        # =====================================================================
        # Sampling phase
        # =====================================================================
        for bi in tqdm(range(num_batches_per_epoch), desc=f"Epoch {epoch}: sampling", disable=not is_main_process(rank)):
            prompts, metas = next(train_iter)
            reward_inputs = []
            local_samples = []

            attn_collector = None
            if use_attn_sync:
                attn_collector = V2AAttentionCollector()
                attn_collector.register_hooks(_unwrap_model(model_engine))

            for i, prompt in enumerate(prompts):
                seed = int(_cfg_get(config, "seed", 42)) + rank * 100000 + epoch * 1000 + bi * 10 + i
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", enabled=enable_amp, dtype=mp_dtype):
                        policy_model = model_engine if use_lora else old_model
                        out = sample_one(sampler, policy_model, prompt, runtime_train, seed, device, attn_collector)

                v2a_vw, v2a_aw = None, None
                vshape = VideoLatentShape.from_torch_shape(out["video_latent"].shape)
                ashape = AudioLatentShape.from_torch_shape(out["audio_latent"].shape)
                if attn_collector is not None:
                    if attn_sync_w_max > 1.0:
                        if attn_sync_warmup_steps > 0:
                            progress = min(global_step / attn_sync_warmup_steps, 1.0)
                            cur_attn_sync_w = 1.0 + (attn_sync_w_max - 1.0) * progress
                        elif attn_sync_warmup_frac > 0:
                            progress = min(global_step / (attn_sync_warmup_frac * total_steps_for_schedule), 1.0)
                            cur_attn_sync_w = 1.0 + (attn_sync_w_max - 1.0) * progress
                        else:
                            cur_attn_sync_w = attn_sync_w_max
                        v2a_vw, v2a_aw = attn_collector.get_weights(cur_attn_sync_w, device, num_frames=vshape.frames)
                    attn_collector.reset()

                v_tools = VideoLatentTools(video_patchifier, vshape, fps=runtime_train.frame_rate)
                a_tools = AudioLatentTools(audio_patchifier, ashape)
                model_dtype = next(_unwrap_model(model_engine).parameters()).dtype

                v_state = v_tools.create_initial_state(device=device, dtype=model_dtype, initial_latent=out["video_latent"].to(device))
                a_state = a_tools.create_initial_state(device=device, dtype=model_dtype, initial_latent=out["audio_latent"].to(device))

                item = {
                    "prompt": prompt,
                    "meta": metas[i],
                    "video_x0": v_state.clean_latent,
                    "video_mask": v_state.denoise_mask,
                    "video_pos": v_state.positions,
                    "audio_x0": a_state.clean_latent,
                    "audio_mask": a_state.denoise_mask,
                    "audio_pos": a_state.positions,
                    "v_ctx": out["v_context"].to(device),
                    "a_ctx": out["a_context"].to(device),
                    "sigmas": sigmas_train.clone(),
                    "temp_video_path": None,
                    "temp_audio_path": None,
                    "v2a_video_weight": v2a_vw,
                    "v2a_audio_weight": v2a_aw,
                }

                need_disk_media = use_video_reward or should_save
                if need_disk_media:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                        path = f.name
                        path = os.path.join(temp_dir, os.path.basename(path))
                    encode_video(out["video"], int(round(runtime_train.frame_rate)), out["audio"], path, 1)
                    item["temp_video_path"] = path

                    audio_obj = out["audio"]
                    if audio_obj is not None:
                        wav_path = os.path.splitext(path)[0] + ".wav"
                        wav = audio_obj.waveform if isinstance(audio_obj, Audio) else audio_obj
                        wav_sr = audio_obj.sampling_rate if isinstance(audio_obj, Audio) else int(_cfg_get(config, "sample.audio_fps", 16000))
                        if wav is not None:
                            if wav.dim() == 1:
                                wav = wav.unsqueeze(0)
                            elif wav.dim() == 2:
                                if wav.shape[0] > wav.shape[1]:
                                    wav = wav.transpose(0, 1)
                            else:
                                raise ValueError(f"Unexpected audio shape: {wav.shape}")
                            torchaudio.save(wav_path, wav.detach().cpu(), sample_rate=int(wav_sr))
                            item["temp_audio_path"] = wav_path

                    if use_video_reward:
                        reward_inputs.append(path)

                if not use_video_reward:
                    reward_inputs.append(out["video"][0].permute(2, 0, 1).float() / 255.0)

                local_samples.append(item)

            reward_input = reward_inputs if use_video_reward else torch.stack(reward_inputs).to(device)
            rewards, _ = reward_fn(reward_input, prompts, metas, only_strict=True)

            for i, item in enumerate(local_samples):
                item["rewards"] = {k: v[i] for k, v in rewards.items() if k != "avg"}
                item["reward"] = rewards["avg"][i]

                temp_video_path = item.pop("temp_video_path", None)
                temp_audio_path = item.pop("temp_audio_path", None)

                if temp_video_path is not None and os.path.exists(temp_video_path):
                    if should_save and sample_save_dir is not None:
                        rew_val = float(rewards["avg"][i])
                        safe_prompt = _safe_prompt_name(item["prompt"], max_len=15)
                        ext = os.path.splitext(temp_video_path)[1]
                        filename = f"rank{rank}_b{bi}_s{i}_rew_{rew_val:.3f}_{safe_prompt}{ext}"
                        final_save_path = os.path.join(sample_save_dir, filename)
                        try:
                            shutil.move(temp_video_path, final_save_path)
                        except Exception as e:
                            logger.warning(f"Failed to move temp file {temp_video_path} to {final_save_path}: {e}")

                        if temp_audio_path is not None and os.path.exists(temp_audio_path):
                            final_wav_path = os.path.splitext(final_save_path)[0] + ".wav"
                            try:
                                shutil.move(temp_audio_path, final_wav_path)
                            except Exception as e:
                                logger.warning(f"Failed to move temp wav {temp_audio_path} to {final_wav_path}: {e}")
                    else:
                        os.remove(temp_video_path)
                        if temp_audio_path is not None and os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)

                samples.append(item)

            if attn_collector is not None:
                attn_collector.remove_hooks()
                attn_collector = None

        if len(samples) == 0:
            continue

        # =====================================================================
        # Advantage computation
        # =====================================================================
        prompts_all = gather_list_to_all([s["prompt"] for s in samples], world_size)

        gathered_rewards_dict = {}
        reward_config = _cfg_get(config, "reward_fn", {})

        rewards_local = torch.tensor([s["reward"] for s in samples], device=device).float()
        gathered_rewards_dict["avg"] = gather_tensor_to_all(rewards_local, world_size).numpy()

        def _is_metric_reward(name: str) -> bool:
            return name in ["avg", "accuracy", "strict_accuracy"] or name.endswith("_accuracy") or name.endswith("_strict_accuracy")

        reward_keys = [k for k in reward_config.keys() if not _is_metric_reward(k)]
        for reward_name in reward_keys:
            if "rewards" in samples[0] and reward_name in samples[0]["rewards"]:
                rewards_local = torch.tensor([s["rewards"][reward_name] for s in samples], device=device).float()
                gathered_rewards_dict[reward_name] = gather_tensor_to_all(rewards_local, world_size).numpy()

        def _as_2d_np(x):
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = np.expand_dims(x, axis=-1)
            return x

        def _adv_from_rewards(rewards_2d):
            rewards_2d = _as_2d_np(rewards_2d)
            if _cfg_get(config, "per_prompt_stat_tracking", True):
                adv = stat_tracker.update(prompts_all, rewards_2d)
                stat_tracker.clear()
                return adv
            return (rewards_2d - rewards_2d.mean(axis=0, keepdims=True)) / (rewards_2d.std(axis=0, keepdims=True) + 1e-4)

        adv_mode = str(_cfg_get(config, "train.adv_mode", "grpo")).lower()

        avg_rewards_2d = _as_2d_np(gathered_rewards_dict["avg"])
        avg_adv_all = _adv_from_rewards(avg_rewards_2d)

        local_n = len(samples)
        num_v_tokens = samples[0]["video_x0"].shape[1]
        num_a_tokens = samples[0]["audio_x0"].shape[1]

        def _extract_local_adv(adv_np):
            return torch.from_numpy(adv_np.reshape(world_size, local_n, -1)[rank][:, 0]).to(device=device, dtype=torch.float32)

        if use_attn_sync and samples[0]["v2a_video_weight"] is not None:
            v2a_vw = torch.cat([s["v2a_video_weight"] for s in samples], dim=0)[:, :, 0]
        else:
            v2a_vw = torch.ones(local_n, num_v_tokens, device=device)
        if use_attn_sync and samples[0]["v2a_audio_weight"] is not None:
            v2a_aw = torch.cat([s["v2a_audio_weight"] for s in samples], dim=0)[:, :, 0]
        else:
            v2a_aw = torch.ones(local_n, num_a_tokens, device=device)

        if adv_mode == "gdpo":
            shared_adv_all = None
            for reward_name in reward_keys:
                if reward_name not in gathered_rewards_dict:
                    continue
                reward_adv = _adv_from_rewards(gathered_rewards_dict[reward_name])
                weight = float(reward_config.get(reward_name, 1.0))
                weighted_adv = weight * reward_adv
                if shared_adv_all is None:
                    shared_adv_all = weighted_adv
                else:
                    shared_adv_all = shared_adv_all + weighted_adv

            if shared_adv_all is None:
                logger.warning("adv_mode=gdpo but no component rewards found, fallback to avg reward advantage.")
                shared_adv_all = avg_adv_all

            shared_local = _extract_local_adv(shared_adv_all)
            adv_video_per_token = shared_local[:, None].expand(-1, num_v_tokens)
            adv_audio_per_token = shared_local[:, None].expand(-1, num_a_tokens)

        elif adv_mode == "branch_aware":
            video_route_keys = _normalize_route_keys(_cfg_get(config, "reward_route.video_keys", ["hpsv3_score_video", "videoalign_score"]))
            audio_route_keys = _normalize_route_keys(_cfg_get(config, "reward_route.audio_keys", ["audiobox_aesthetics_score", "clap_score"]))
            sync_route_keys = _normalize_route_keys(_cfg_get(config, "reward_route.sync_keys", ["av_align_score", "av_desync_reward"]))

            per_reward_adv = {}
            for reward_name in reward_keys:
                if reward_name not in gathered_rewards_dict:
                    continue
                rewards_2d = _as_2d_np(gathered_rewards_dict[reward_name])
                per_reward_adv[reward_name] = _adv_from_rewards(rewards_2d)

            adv_video_all = None
            adv_audio_all = None
            for reward_name, adv in per_reward_adv.items():
                weight = float(reward_config.get(reward_name, 1.0))
                route = _infer_reward_route(reward_name, video_route_keys, audio_route_keys, sync_route_keys)
                weighted_adv = weight * adv

                if route == "video":
                    adv_video_all = weighted_adv if adv_video_all is None else adv_video_all + weighted_adv
                elif route == "audio":
                    adv_audio_all = weighted_adv if adv_audio_all is None else adv_audio_all + weighted_adv
                elif route == "sync":
                    adv_video_all = weighted_adv if adv_video_all is None else adv_video_all + weighted_adv
                    adv_audio_all = weighted_adv if adv_audio_all is None else adv_audio_all + weighted_adv

            if adv_video_all is None or adv_video_all.sum() == 0:
                adv_video_all = avg_adv_all
            if adv_audio_all is None or adv_audio_all.sum() == 0:
                adv_audio_all = avg_adv_all

            adv_video_local = _extract_local_adv(adv_video_all)
            adv_audio_local = _extract_local_adv(adv_audio_all)
            adv_video_per_token = adv_video_local[:, None].expand(-1, num_v_tokens)
            adv_audio_per_token = adv_audio_local[:, None].expand(-1, num_a_tokens)

        else:
            raise ValueError(f"Unsupported train.adv_mode={adv_mode}. Supported: gdpo, branch_aware")

        packed = {
            "video_x0": torch.cat([s["video_x0"] for s in samples], dim=0),
            "video_mask": torch.cat([s["video_mask"] for s in samples], dim=0),
            "video_pos": torch.cat([s["video_pos"] for s in samples], dim=0),
            "audio_x0": torch.cat([s["audio_x0"] for s in samples], dim=0),
            "audio_mask": torch.cat([s["audio_mask"] for s in samples], dim=0),
            "audio_pos": torch.cat([s["audio_pos"] for s in samples], dim=0),
            "v_ctx": torch.cat([s["v_ctx"] for s in samples], dim=0),
            "a_ctx": torch.cat([s["a_ctx"] for s in samples], dim=0),
            "sigmas": torch.stack([s["sigmas"] for s in samples], dim=0),
            "adv_video": adv_video_per_token,
            "adv_audio": adv_audio_per_token,
            "attn_w_video": v2a_vw,
            "attn_w_audio": v2a_aw,
        }

        # =====================================================================
        # Training phase
        # =====================================================================
        model_engine.train()
        if ca_balancer is not None:
            ca_balancer.enabled = True
            ca_balancer.set_progress(global_step / max(num_epochs * num_batches_per_epoch, 1))
        if use_lora:
            _set_adapter(model_engine, "default")
        accum_steps = 0

        for inner in range(num_inner):
            perm = torch.randperm(local_n, device=device)
            micro_batches = [perm[i: i + train_bs] for i in range(0, local_n, train_bs)]
            info = defaultdict(list)

            for mb_idx in micro_batches:
                b = {k: v[mb_idx] for k, v in packed.items()}

                for t_idx in tqdm(torch.randperm(num_train_t, device=device)):
                    next_step = accum_steps + 1
                    is_sync_step = (next_step % eff_accum == 0)

                    sigma = b["sigmas"][:, t_idx].float()
                    sigma_token = sigma[:, None, None]

                    x0_v = b["video_x0"].float()
                    x0_a = b["audio_x0"].float()
                    sigma_v = b["video_mask"].float() * sigma_token
                    sigma_a = b["audio_mask"].float() * sigma_token
                    xt_v = (1 - sigma_v) * x0_v + sigma_v * torch.randn_like(x0_v)
                    xt_a = (1 - sigma_a) * x0_a + sigma_a * torch.randn_like(x0_a)

                    model_dtype = next(_unwrap_model(model_engine).parameters()).dtype
                    mod_v = Modality(
                        latent=xt_v.to(model_dtype),
                        sigma=sigma,
                        timesteps=(b["video_mask"].float() * sigma_token).to(model_dtype),
                        positions=b["video_pos"],
                        context=b["v_ctx"],
                    )
                    mod_a = Modality(
                        latent=xt_a.to(model_dtype),
                        sigma=sigma,
                        timesteps=(b["audio_mask"].float() * sigma_token).to(model_dtype),
                        positions=b["audio_pos"],
                        context=b["a_ctx"],
                    )

                    with torch.autocast(device_type="cuda", enabled=enable_amp, dtype=mp_dtype):
                        if use_lora:
                            with torch.no_grad():
                                _set_adapter(model_engine, "old")
                                old_vel_v, old_vel_a = model_engine(video=mod_v, audio=mod_a, perturbations=None)
                                old_pred_v = to_denoised(xt_v.to(old_vel_v.dtype), old_vel_v, sigma_v.to(old_vel_v.dtype)).detach()
                                old_pred_a = to_denoised(xt_a.to(old_vel_a.dtype), old_vel_a, sigma_a.to(old_vel_a.dtype)).detach()

                                _set_adapter(model_engine, "default")
                                with _disable_adapter_ctx(model_engine):
                                    ref_vel_v, ref_vel_a = model_engine(video=mod_v, audio=mod_a, perturbations=None)
                                ref_pred_v = to_denoised(xt_v.to(ref_vel_v.dtype), ref_vel_v, sigma_v.to(ref_vel_v.dtype)).detach()
                                ref_pred_a = to_denoised(xt_a.to(ref_vel_a.dtype), ref_vel_a, sigma_a.to(ref_vel_a.dtype)).detach()
                                ref_vel_v = ref_vel_v.detach()
                                ref_vel_a = ref_vel_a.detach()

                            _set_adapter(model_engine, "default")
                            fwd_vel_v, fwd_vel_a = model_engine(video=mod_v, audio=mod_a, perturbations=None)
                            fwd_pred_v = to_denoised(xt_v.to(fwd_vel_v.dtype), fwd_vel_v, sigma_v.to(fwd_vel_v.dtype))
                            fwd_pred_a = to_denoised(xt_a.to(fwd_vel_a.dtype), fwd_vel_a, sigma_a.to(fwd_vel_a.dtype))
                        else:
                            with torch.no_grad():
                                old_vel_v, old_vel_a = old_model(video=mod_v, audio=mod_a, perturbations=None)
                                ref_vel_v, ref_vel_a = ref_model(video=mod_v, audio=mod_a, perturbations=None)
                                old_pred_v = to_denoised(xt_v.to(old_vel_v.dtype), old_vel_v, sigma_v.to(old_vel_v.dtype)).detach()
                                old_pred_a = to_denoised(xt_a.to(old_vel_a.dtype), old_vel_a, sigma_a.to(old_vel_a.dtype)).detach()
                                ref_pred_v = to_denoised(xt_v.to(ref_vel_v.dtype), ref_vel_v, sigma_v.to(ref_vel_v.dtype)).detach()
                                ref_pred_a = to_denoised(xt_a.to(ref_vel_a.dtype), ref_vel_a, sigma_a.to(ref_vel_a.dtype)).detach()
                                ref_vel_v = ref_vel_v.detach()
                                ref_vel_a = ref_vel_a.detach()
                            fwd_vel_v, fwd_vel_a = model_engine(video=mod_v, audio=mod_a, perturbations=None)
                            fwd_pred_v = to_denoised(xt_v.to(fwd_vel_v.dtype), fwd_vel_v, sigma_v.to(fwd_vel_v.dtype))
                            fwd_pred_a = to_denoised(xt_a.to(fwd_vel_a.dtype), fwd_vel_a, sigma_a.to(fwd_vel_a.dtype))

                        adv_clip_max = float(_cfg_get(config, "train.adv_clip_max", 5.0))
                        beta_mix = float(_cfg_get(config, "beta", 1.0))
                        modality_weights = {
                            "video": float(_cfg_get(config, "train.video_loss_weight", 1.0)),
                            "audio": float(_cfg_get(config, "train.audio_loss_weight", 1.0)),
                        }

                        modality_states = {
                            "video": {"fwd": fwd_pred_v, "old": old_pred_v, "ref": ref_pred_v, "fwd_vel": fwd_vel_v, "ref_vel": ref_vel_v, "xt": xt_v, "x0": x0_v, "sigma_cur": mod_v.timesteps.float()},
                            "audio": {"fwd": fwd_pred_a, "old": old_pred_a, "ref": ref_pred_a, "fwd_vel": fwd_vel_a, "ref_vel": ref_vel_a, "xt": xt_a, "x0": x0_a, "sigma_cur": mod_a.timesteps.float()},
                        }
                        modality_adv_key = {"video": "adv_video", "audio": "adv_audio"}

                        modality_policy = {}
                        modality_kl = {}
                        for modality in ["video", "audio"]:
                            state = modality_states[modality]
                            x0_cur = state["x0"]
                            xt_cur = state["xt"]
                            sigma_cur = state["sigma_cur"]

                            fwd_pred = state["fwd"]
                            old_pred = state["old"]
                            ref_pred = state["ref"]

                            pos_pred = beta_mix * fwd_pred + (1 - beta_mix) * old_pred
                            neg_pred = (1 + beta_mix) * old_pred - beta_mix * fwd_pred

                            x0_pos = pos_pred
                            x0_neg = neg_pred

                            # x0_pos = xt_cur - sigma_cur * pos_pred
                            # x0_neg = xt_cur - sigma_cur * neg_pred

                            reduce_dims = tuple(range(1, x0_cur.ndim))

                            wp = (x0_pos.double() - x0_cur.double()).abs().mean(dim=reduce_dims, keepdim=True).clip(min=1e-5)
                            wn = (x0_neg.double() - x0_cur.double()).abs().mean(dim=reduce_dims, keepdim=True).clip(min=1e-5)

                            adv_per_token = torch.clamp(b[modality_adv_key[modality]], -adv_clip_max, adv_clip_max)
                            r_token = torch.clamp((adv_per_token / adv_clip_max) / 2.0 + 0.5, 0.0, 1.0)

                            pos_per_token = ((x0_pos - x0_cur) ** 2 / wp).mean(dim=-1)
                            neg_per_token = ((x0_neg - x0_cur) ** 2 / wn).mean(dim=-1)
                            policy_per_token = r_token * pos_per_token + (1 - r_token) * neg_per_token

                            if modality == "video":
                                attn_w = b["attn_w_video"]
                            else:
                                attn_w = torch.ones_like(policy_per_token)

                            policy_m = (policy_per_token * attn_w).sum(dim=1) / attn_w.sum(dim=1) / beta_mix * adv_clip_max
                            # policy_m = (policy_per_token * attn_w).mean(dim=1) / beta_mix * adv_clip_max
                            
                            kl_per_token = ((state["fwd"] - state["ref"]) ** 2).mean(dim=-1)
                            kl_m = kl_per_token.mean(dim=1)

                            modality_policy[modality] = policy_m
                            modality_kl[modality] = kl_m

                        weight_sum = max(sum(modality_weights.values()), 1e-8)
                        policy = (modality_weights["video"] * modality_policy["video"]
                                  + modality_weights["audio"] * modality_policy["audio"]) / weight_sum
                        kl = (modality_weights["video"] * modality_kl["video"]
                              + modality_weights["audio"] * modality_kl["audio"]) / weight_sum
                        loss = policy.mean() + float(_cfg_get(config, "train.beta", 1e-4)) * kl.mean()

                    loss_scaled = loss / eff_accum

                    # Backward (FSDP vs DDP)
                    if use_fsdp:
                        sync_ctx = nullcontext() if is_sync_step else model_engine.no_sync()
                        with sync_ctx:
                            loss_scaled.backward()
                    else:
                        if scaler.is_enabled():
                            scaler.scale(loss_scaled).backward()
                        else:
                            loss_scaled.backward()

                    accum_steps = next_step
                    info["loss"].append(loss.detach())
                    info["policy"].append(policy.mean().detach())
                    info["policy_v"].append(modality_policy["video"].mean().detach())
                    info["policy_a"].append(modality_policy["audio"].mean().detach())
                    info["kl"].append(kl.mean().detach())
                    info["kl_v"].append(modality_kl["video"].mean().detach())
                    info["kl_a"].append(modality_kl["audio"].mean().detach())

                    if is_sync_step:
                        max_grad_norm = float(_cfg_get(config, "train.max_grad_norm", 1.0))
                        if use_fsdp:
                            if hasattr(model_engine, "clip_grad_norm_"):
                                model_engine.clip_grad_norm_(max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                            optimizer.step()
                        else:
                            if scaler.is_enabled():
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                            if scaler.is_enabled():
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                        if ema is not None:
                            ema.step(trainable_params, global_step)

                        stats = {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}
                        keys = sorted(stats.keys())
                        t = torch.tensor([stats[k] for k in keys], device=device)
                        dist.all_reduce(t, op=dist.ReduceOp.AVG)

                        if is_main_process(rank) and writer is not None:
                            reduced_stats = {k: t[i].item() for i, k in enumerate(keys)}
                            for key, value in reduced_stats.items():
                                writer.add_scalar(f"Loss/{key}", value, global_step=global_step)
                            writer.add_scalar("Meta/epoch", float(epoch), global_step=global_step)
                            writer.add_scalar("Meta/inner_epoch", float(inner), global_step=global_step)

                            if use_attn_sync:
                                writer.add_scalar("Meta/v2a_attn_weight_mean", v2a_vw.mean().item(), global_step=global_step)
                                writer.add_scalar("Meta/v2a_attn_weight_max", v2a_vw.max().item(), global_step=global_step)
                                if attn_sync_warmup_steps > 0:
                                    progress = min(global_step / attn_sync_warmup_steps, 1.0)
                                    writer.add_scalar("Meta/attn_sync_w_scheduled", 1.0 + (attn_sync_w_max - 1.0) * progress, global_step=global_step)
                                elif attn_sync_warmup_frac > 0:
                                    progress = min(global_step / (attn_sync_warmup_frac * total_steps_for_schedule), 1.0)
                                    writer.add_scalar("Meta/attn_sync_w_scheduled", 1.0 + (attn_sync_w_max - 1.0) * progress, global_step=global_step)

                            gathered_rewards_dict["avg"] = gathered_rewards_dict["avg"].reshape(-1, 1)
                            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all, gathered_rewards_dict)
                            group_size, trained_prompt_num = stat_tracker.get_stats()
                            metrics = {
                                "rewards/group_size": group_size,
                                "rewards/trained_prompt_num": trained_prompt_num,
                                "rewards/zero_std_ratio": zero_std_ratio,
                                "rewards/reward_std_mean": reward_std_mean,
                                "rewards/mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                            }
                            for key, value in metrics.items():
                                writer.add_scalar(key, value, global_step=global_step)

                            for k, v in gathered_rewards_dict.items():
                                if "_strict_accuracy" not in k and "_accuracy" not in k:
                                    writer.add_scalar(f"reward_score/{k}", v.mean(), global_step=global_step)

                        info = defaultdict(list)

        # =====================================================================
        # Old policy EMA update
        # =====================================================================
        if use_lora:
            if use_fsdp:
                with FSDP.summon_full_params(model_engine, writeback=True):
                    with torch.no_grad():
                        decay = return_decay(global_step, int(_cfg_get(config, "decay_type", 1)))
                        fsdp_params = dict(model_engine.named_parameters())
                        for name, src_param in fsdp_params.items():
                            if "default" in name and src_param.requires_grad:
                                old_name = name.replace("default", "old")
                                if old_name in fsdp_params:
                                    tgt = fsdp_params[old_name]
                                    tgt.copy_(tgt.detach() * decay + src_param.detach() * (1.0 - decay))
            else:
                with torch.no_grad():
                    decay = return_decay(global_step, int(_cfg_get(config, "decay_type", 1)))
                    _set_adapter(model_engine, "default")
                    src_params = _collect_trainable_params(model_engine)
                    _set_adapter(model_engine, "old")
                    tgt_params = _collect_trainable_params(model_engine)
                    for src, tgt in zip(src_params, tgt_params, strict=True):
                        tgt.data.copy_(tgt.data * decay + src.data * (1 - decay))
                    _set_adapter(model_engine, "default")
        else:
            with torch.no_grad():
                decay = return_decay(global_step, int(_cfg_get(config, "decay_type", 1)))
                if use_fsdp:
                    with FSDP.summon_full_params(model_engine, recurse=True, writeback=False):
                        with FSDP.summon_full_params(old_model, recurse=True, writeback=True):
                            src_params = list(_unwrap_model(model_engine).parameters())
                            tgt_params = list(_unwrap_model(old_model).parameters())
                            for src, tgt in zip(src_params, tgt_params, strict=True):
                                tgt.data.copy_(tgt.data * decay + src.data * (1 - decay))
                else:
                    for src, tgt in zip(_unwrap_model(model_engine).parameters(), old_model.parameters(), strict=True):
                        tgt.data.copy_(tgt.data * decay + src.data * (1 - decay))

        # =====================================================================
        # Checkpoint saving
        # =====================================================================
        if (epoch % save_freq == 0) and not debug:
            save_ckpt(log_dir, model_engine, optimizer, scaler, global_step, rank, trainable_params, ema,
                      use_fsdp=use_fsdp, use_lora=use_lora, epoch=epoch)

        # =====================================================================
        # Evaluation
        # =====================================================================
        if (epoch % eval_freq == 0) and not debug:
            if ca_balancer is not None:
                ca_balancer.enabled = False
            eval_scores = []
            save_eval_videos = bool(_cfg_get(config, "sample.save_eval_videos", True))
            eval_save_dir = os.path.join(log_dir, "eval_samples", f"step_{global_step}")
            if save_eval_videos:
                os.makedirs(eval_save_dir, exist_ok=True)

            for bidx, (prompts, metas) in enumerate(test_loader):
                reward_inputs = []
                for i, prompt in enumerate(prompts):
                    seed = int(_cfg_get(config, "seed", 42)) + 999999 + epoch * 1000 + rank * 100 + i
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", enabled=enable_amp, dtype=mp_dtype):
                            eval_model = model_engine
                            if use_lora:
                                _set_adapter(model_engine, "default")
                            out = sample_one(sampler, eval_model, prompt, runtime_eval, seed, device)

                    saved_video_path = None
                    if save_eval_videos:
                        safe_prompt = _safe_prompt_name(prompt, max_len=15)
                        saved_video_path = os.path.join(
                            eval_save_dir,
                            f"b{bidx}_sample_{i}_rank_{rank}_{safe_prompt}.mp4",
                        )
                        encode_video(out["video"], int(round(runtime_eval.frame_rate)), out["audio"], saved_video_path, 1)

                        audio_obj = out["audio"]
                        if audio_obj is not None:
                            wav_path = os.path.splitext(saved_video_path)[0] + ".wav"
                            wav = audio_obj.waveform if isinstance(audio_obj, Audio) else audio_obj
                            wav_sr = audio_obj.sampling_rate if isinstance(audio_obj, Audio) else int(_cfg_get(config, "sample.audio_fps", 16000))
                            if wav is not None:
                                if wav.dim() == 1:
                                    wav = wav.unsqueeze(0)
                                elif wav.dim() == 2:
                                    if wav.shape[0] > wav.shape[1]:
                                        wav = wav.transpose(0, 1)
                                else:
                                    raise ValueError(f"Unexpected audio shape: {wav.shape}")
                                torchaudio.save(wav_path, wav.detach().cpu(), sample_rate=int(wav_sr))

                    if use_video_reward:
                        if saved_video_path is not None:
                            reward_inputs.append(saved_video_path)
                        else:
                            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                                path = f.name
                            encode_video(out["video"], int(round(runtime_eval.frame_rate)), out["audio"], path, 1)
                            reward_inputs.append(path)
                    else:
                        reward_inputs.append(out["video"][0].permute(2, 0, 1).float() / 255.0)

                if len(reward_inputs) == 0:
                    continue

                reward_input = reward_inputs if use_video_reward else torch.stack(reward_inputs).to(device)
                rewards, _ = reward_fn(reward_input, prompts, metas, only_strict=False)
                eval_scores.append(torch.as_tensor(rewards["avg"], device=device).float())

                if use_video_reward and (not save_eval_videos):
                    for _p in reward_inputs:
                        if os.path.exists(_p):
                            os.remove(_p)

            if len(eval_scores) > 0:
                eval_score = torch.cat(eval_scores, dim=0).mean()
                eval_score_all = gather_tensor_to_all(eval_score[None], world_size)
                if is_main_process(rank) and writer is not None:
                    writer.add_scalar("Eval/eval_reward_avg", float(eval_score_all.mean().item()), global_step=global_step)

        if world_size > 1:
            dist.barrier()

    if is_main_process(rank) and writer is not None:
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
