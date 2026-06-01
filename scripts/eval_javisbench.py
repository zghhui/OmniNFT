import argparse
import csv
import json
import logging
import os
from pathlib import Path

import torch

import sys
from safetensors.torch import save_file

_SCRIPT_DIR = Path(__file__).resolve().parent
_LTX_V1_ROOT = _SCRIPT_DIR.parent / "ltx_v1"
# 直接加 ltx_v1 根目录即可
if str(_LTX_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_LTX_V1_ROOT))

        
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
from ltx_pipelines.utils.media_io import encode_video


def parse_args():
    parser = argparse.ArgumentParser("LTX-2 multi-node multi-gpu inference from CSV")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--gemma-root", type=str, required=True)

    # prompt 输入
    parser.add_argument("--dataset-path", type=str, default=None, help="数据集目录或文件路径")
    parser.add_argument("--prompt-path", type=str, default=None, help="直接指定 prompt 文件(.csv/.jsonl/.txt)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt-key", type=str, default="prompt")

    # 切片范围
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)

    # 输出
    parser.add_argument("--output-dir", type=str, required=True)

    # seed
    parser.add_argument("--seed", type=int, default=-1, help="-1 随机；>=0 固定")
    return parser.parse_args()


def _build_infer_config_from_model(checkpoint_path: str) -> dict:
    params = detect_params(checkpoint_path)
    video_guider_default = params.video_guider_params
    audio_guider_default = params.audio_guider_params

    return {
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "seed": int(params.seed),
        "height": int(params.stage_1_height),
        "width": int(params.stage_1_width),
        "num_frames": int(params.num_frames),
        "frame_rate": float(params.frame_rate),
        "num_inference_steps": int(params.num_inference_steps),
        "video_guider_params": MultiModalGuiderParams(
            cfg_scale=video_guider_default.cfg_scale,
            stg_scale=video_guider_default.stg_scale,
            rescale_scale=video_guider_default.rescale_scale,
            modality_scale=video_guider_default.modality_scale,
            skip_step=video_guider_default.skip_step,
            stg_blocks=video_guider_default.stg_blocks,
        ),
        "audio_guider_params": MultiModalGuiderParams(
            cfg_scale=audio_guider_default.cfg_scale,
            stg_scale=audio_guider_default.stg_scale,
            rescale_scale=audio_guider_default.rescale_scale,
            modality_scale=audio_guider_default.modality_scale,
            skip_step=audio_guider_default.skip_step,
            stg_blocks=audio_guider_default.stg_blocks,
        ),
        "enhance_prompt": False,
    }


def _extract_prompt(item: dict, prompt_key: str) -> str:
    if prompt_key in item and isinstance(item[prompt_key], str) and item[prompt_key].strip():
        return item[prompt_key]
    for key in ("prompt_av", "prompt", "text", "caption", "prompt_v", "prompt_a"):
        value = item.get(key, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _read_prompts_from_file(path: Path, prompt_key: str) -> list[dict]:
    samples: list[dict] = []
    suffix = path.suffix.lower()

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    samples.append({"prompt": prompt, "sample_idx": len(samples)})
        return samples

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = item if isinstance(item, str) else _extract_prompt(item, prompt_key)
                if isinstance(prompt, str) and prompt.strip():
                    samples.append({"prompt": prompt, "sample_idx": len(samples)})
        return samples

    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = _extract_prompt(row, prompt_key)
                if prompt.strip():
                    samples.append({"prompt": prompt, "sample_idx": len(samples)})
        return samples

    # fallback: 按 txt 读取
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                samples.append({"prompt": prompt, "sample_idx": len(samples)})
    return samples


def _resolve_prompt_file(dataset_path: str | None, prompt_path: str | None, split: str) -> Path:
    if prompt_path:
        p = Path(prompt_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"prompt 文件不存在: {p}")
        return p

    if not dataset_path:
        raise ValueError("请提供 --dataset-path 或 --prompt-path")

    d = Path(dataset_path).expanduser().resolve()
    if d.is_file():
        return d

    candidates = [d / f"{split}.csv", d / f"{split}_metadata.jsonl", d / f"{split}.jsonl", d / f"{split}.txt"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"在数据集目录中未找到可用 prompt 文件: {d}")


def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


@torch.inference_mode()
def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    rank, world_size, local_rank = setup_distributed()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = _resolve_prompt_file(args.dataset_path, args.prompt_path, args.split)
    all_samples = _read_prompts_from_file(prompt_file, args.prompt_key)
    selected_samples = (
        all_samples[args.start_index:] if args.end_index is None
        else all_samples[args.start_index: args.end_index]
    )
    rank_samples = [sample for i, sample in enumerate(selected_samples) if i % world_size == rank]

    logging.info(
        f"[Rank {rank}] world_size={world_size}, total={len(all_samples)}, "
        f"selected={len(selected_samples)}, this_rank={len(rank_samples)}"
    )

    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=(),
        quantization=None,
    )
    infer_config = _build_infer_config_from_model(args.checkpoint_path)

    for i, sample in enumerate(rank_samples):
        prompt = sample["prompt"].strip()
        sample_idx = sample["sample_idx"]

        if not prompt:
            continue

        # seed 策略
        if args.seed < 0:
            run_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        else:
            run_seed = int(args.seed + sample_idx)  # 保证不同样本可复现且不完全一样

        filename = f"sample_{sample_idx:04d}.mp4"
        output_path = output_dir / filename
        if output_path.exists():
            continue

        try:
            video, audio = pipeline(
                prompt=prompt,
                negative_prompt=infer_config["negative_prompt"],
                seed=run_seed,
                height=infer_config["height"],
                width=infer_config["width"],
                num_frames=infer_config["num_frames"],
                frame_rate=infer_config["frame_rate"],
                num_inference_steps=infer_config["num_inference_steps"],
                video_guider_params=infer_config["video_guider_params"],
                audio_guider_params=infer_config["audio_guider_params"],
                images=[],
                enhance_prompt=infer_config["enhance_prompt"],
            )

            encode_video(
                video=video,
                fps=int(infer_config["frame_rate"]),
                audio=audio,
                output_path=str(output_path),
                video_chunks_number=1,
            )
            logging.info(f"[Rank {rank}] done {i+1}/{len(rank_samples)} -> {output_path.name}")
        except Exception as e:
            logging.exception(f"[Rank {rank}] failed sample_idx={sample_idx}, err={e}")

    logging.info(f"[Rank {rank}] All done.")


if __name__ == "__main__":
    main()
