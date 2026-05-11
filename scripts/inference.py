"""Inference script for OmniNFT-LTX: generate audio-video from text prompts.

Usage:
    # Step 1: Merge LoRA into base checkpoint
    python scripts/merge_lora.py \
        --checkpoint-path $LTX_MODEL_PATH \
        --lora-dir $OUTPUT_DIR/checkpoint-latest/lora \
        --output-path ./merged_model.safetensors

    # Step 2: Run inference with merged model
    python scripts/inference.py \
        --model_path ./merged_model.safetensors \
        --prompt "A man plays acoustic guitar on stage"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio

_LTX_V2_ROOT = Path(__file__).resolve().parents[1] / "ltx_v2"
for _pkg in ("ltx-core", "ltx-pipelines", "ltx-trainer"):
    _src = str(_LTX_V2_ROOT / _pkg / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
_rl_core_root = str(_LTX_V2_ROOT / "rl-core")
if _rl_core_root not in sys.path:
    sys.path.insert(0, _rl_core_root)

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.types import Audio
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
from ltx_pipelines.utils.media_io import encode_video
from ltx_trainer.model_loader import load_embeddings_processor, load_model
from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler


def parse_args():
    parser = argparse.ArgumentParser(description="OmniNFT-LTX Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to merged model checkpoint (.safetensors)")
    parser.add_argument("--gemma_path", type=str, default=None, help="Path to Gemma text encoder (default: env GEMMA_MODEL_PATH)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of video frames")
    parser.add_argument("--height", type=int, default=None, help="Video height (default: model default)")
    parser.add_argument("--width", type=int, default=None, help="Video width (default: model default)")
    parser.add_argument("--frame_rate", type=float, default=None, help="Frame rate (default: model default)")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Denoising steps (default: model default)")
    parser.add_argument("--video_guidance_scale", type=float, default=None, help="Video CFG scale (default: model default)")
    parser.add_argument("--audio_guidance_scale", type=float, default=None, help="Audio CFG scale (default: model default)")
    parser.add_argument("--stg_scale", type=float, default=None, help="STG scale (default: model default)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference dtype")
    parser.add_argument("--no_audio", action="store_true", help="Disable audio generation")
    return parser.parse_args()


def main():
    args = parse_args()

    gemma_path = args.gemma_path or os.environ.get("GEMMA_MODEL_PATH", "")
    if not gemma_path:
        raise ValueError("Gemma path required: set --gemma_path or env GEMMA_MODEL_PATH")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading merged model from {args.model_path} ...")
    components = load_model(
        checkpoint_path=args.model_path,
        text_encoder_path=gemma_path,
        device="cpu",
        dtype=dtype,
        with_video_vae_encoder=False,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=True,
        with_vocoder=True,
        with_text_encoder=True,
    )
    embeddings_processor = load_embeddings_processor(
        checkpoint_path=args.model_path,
        device="cpu",
        dtype=dtype,
    )

    model = components.transformer.to(device, dtype=dtype).eval()

    sampler = ValidationSampler(
        transformer=model,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=None,
        text_encoder=components.text_encoder,
        embeddings_processor=embeddings_processor,
        audio_decoder=components.audio_vae_decoder,
        vocoder=components.vocoder,
    )

    params = detect_params(args.model_path)
    height = args.height or params.stage_1_height
    width = args.width or params.stage_1_width
    frame_rate = args.frame_rate or params.frame_rate
    num_inference_steps = args.num_inference_steps or params.num_inference_steps
    video_guidance_scale = args.video_guidance_scale or params.video_guider_params.cfg_scale
    audio_guidance_scale = args.audio_guidance_scale or params.audio_guider_params.cfg_scale
    stg_scale = args.stg_scale if args.stg_scale is not None else params.video_guider_params.stg_scale
    stg_blocks = list(params.video_guider_params.stg_blocks)

    gen_config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=int(height),
        width=int(width),
        num_frames=args.num_frames,
        frame_rate=float(frame_rate),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(video_guidance_scale),
        video_guidance_scale=float(video_guidance_scale),
        audio_guidance_scale=float(audio_guidance_scale),
        seed=args.seed,
        condition_image=None,
        generate_audio=not args.no_audio,
        stg_scale=float(stg_scale),
        stg_blocks=stg_blocks if stg_blocks else None,
        stg_mode="stg_av",
    )

    print(f"Generating: {args.prompt}")
    print(f"  Resolution: {int(height)}x{int(width)}, Frames: {args.num_frames}, Steps: {int(num_inference_steps)}")

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(args.dtype != "fp32"), dtype=dtype):
        sampler._validate_config(gen_config)
        v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = sampler._get_prompt_embeddings(gen_config, device)

        video_tools = sampler._create_video_latent_tools(gen_config)
        audio_tools = sampler._create_audio_latent_tools(gen_config)
        generator = torch.Generator(device=device).manual_seed(args.seed)

        video_clean_state = video_tools.create_initial_state(device=device, dtype=dtype)
        audio_clean_state = audio_tools.create_initial_state(device=device, dtype=dtype)

        noiser = GaussianNoiser(generator=generator)
        video_state = noiser(latent_state=video_clean_state, noise_scale=1.0)
        audio_state = noiser(latent_state=audio_clean_state, noise_scale=1.0)

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
        )

        video_unpatch_state = video_tools.clear_conditioning(video_state)
        video_unpatch_state = video_tools.unpatchify(video_unpatch_state)

        audio_unpatch_state = audio_tools.clear_conditioning(audio_state)
        audio_unpatch_state = audio_tools.unpatchify(audio_unpatch_state)

        video_float = sampler._decode_video(video_unpatch_state, device, False)
        video_out = (video_float.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        video_out = video_out.permute(1, 2, 3, 0)

        audio_out = None
        if not args.no_audio:
            audio_waveform = sampler._decode_audio(audio_unpatch_state, device)
            audio_sr = sampler._vocoder.output_sampling_rate if sampler._vocoder is not None else 16000
            audio_out = Audio(waveform=audio_waveform, sampling_rate=audio_sr) if audio_waveform is not None else None

    os.makedirs(args.output_dir, exist_ok=True)

    safe_name = args.prompt[:50].replace(" ", "_").replace("/", "_")
    video_path = os.path.join(args.output_dir, f"{safe_name}_seed{args.seed}.mp4")

    encode_video(video_out, int(round(float(frame_rate))), audio_out, video_path, 1)
    print(f"Video saved: {video_path}")

    if audio_out is not None:
        wav_path = os.path.splitext(video_path)[0] + ".wav"
        wav = audio_out.waveform
        if wav is not None:
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            elif wav.dim() == 2 and wav.shape[0] > wav.shape[1]:
                wav = wav.transpose(0, 1)
            torchaudio.save(wav_path, wav.detach().cpu(), sample_rate=int(audio_out.sampling_rate))
            print(f"Audio saved: {wav_path}")


if __name__ == "__main__":
    main()
