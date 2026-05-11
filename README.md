<h2 align="center">OmniNFT</h2>
<h4 align="center">Modality-wise Omni Diffusion Negative-aware Fine-Tuning</h4>

<p align="center">
  <a href="https://huggingface.co/zghhui/OmniNFT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-OmniNFT-ffc107?logoColor=white" alt="HuggingFace"/></a>
  <a href="https://github.com/zghhui/OmniNFT"><img src="https://img.shields.io/badge/arXiv-Paper-b5212f?logo=arxiv" alt="ArXiv"/></a>
  <a href="https://zghhui.github.io/OmniNFT/"><img src="https://img.shields.io/badge/🌐-Project%20Page-blue" alt="Project Page"/></a>
</p>

---

## 🏗️ Method Overview

**Modality-wise Advantage Routing** — Instead of collapsing all rewards into a single global advantage, OmniNFT computes independent per-reward advantages for video, audio, and cross-modal synchronization, then routes each to its responsible generation branch.

**Layer-wise Gradient Surgery** — OmniNFT applies a partial stop-gradient on the audio key-value projections in A2V cross-attention at shallow Transformer blocks, suppressing erroneous gradient injection while preserving full gradient flow through the deeper cross-modal alignment layers (AV-Sync Zone).

**Region-wise Loss Reweighting** — Leveraging V2A cross-attention maps from late denoising steps as an intrinsic proxy for sound-emitting critical regions, OmniNFT aggregates them into per-token importance weights.

---

## ⚡ Installation

```bash
conda create -n omnninft python=3.11
conda activate omnninft
pip install -r requirements.txt
```

<!-- --- -->

## 📦 Model Checkpoints

| Env Variable | Description | Source |
|---|---|---|
| `LTX-2_MODEL` | LTX-2 base model | [LTX-2](https://huggingface.co/Lightricks/LTX-2) |
| `OmniNFT_LTX-2` | LTX-2 + OmniNFT | [OmniNFT](https://huggingface.co/zghhui/OmniNFT) |
| `REWARD_MODELS` | All reward models (HPSv3, CLAP, AudioBox, Synchformer, ImageBind, etc.) | [OmniNFT-Reward-Series](https://huggingface.co/zghhui/Omni_Reward_Series) |


## 🚀 Training

### Step 1: Launch Reward Servers

HPSv3 and VideoAlign run as remote HTTP servers. Start them **before** training:

```bash
# Terminal 1: HPSv3 server
export HPSV3_CKPT_PATH=/path/to/HPSv3.safetensors
bash flow_grpo/server/run_remote_hpsv3.sh

# Terminal 2: VideoAlign server
export VIDEOALIGN_CKPT_DIR=/path/to/VideoReward
bash flow_grpo/server/run_remote_videoalign.sh
```

### Step 2: Multi(Single)-Node Training (Optional)

Set the following on **each** node:

```bash

export HPSV3_REWARD_SERVER=<server_ip>
export HPSV3_REWARD_PORT=8001
export VIDEOALIGN_REWARD_SERVER=<server_ip>
export VIDEOALIGN_REWARD_PORT=8002

export WORLD_SIZE=<num_nodes>
export RANK=<node_rank>
export MASTER_ADDR=<master_ip>
export MASTER_PORT=6000

bash train_omninft_ltx_fsdp.sh branch_aware_layer_surgery_avweight
```


## 🎬 Inference

### Step 1: Merge LoRA into base model

After training, merge the LoRA weights into the base checkpoint:

```bash
python scripts/merge_lora.py \
    --checkpoint-path $LTX_MODEL_PATH \
    --lora-dir $OUTPUT_DIR/checkpoint-latest/lora \
    --output-path ./merged_model.safetensors \
    --dtype bf16
```

<details>
<summary><strong>Arguments</strong></summary>

| Argument | Description |
|---|---|
| `--checkpoint-path` | LTX-Video base checkpoint used during training |
| `--lora-dir` | LoRA output directory (contains `adapter_model.safetensors` + `adapter_config.json`) |
| `--output-path` | Output path for the merged model |
| `--dtype` | Output precision: `bf16` (default) / `fp16` / `fp32` / `keep` |

</details>

### Step 2: Generate audio-video

```bash
python scripts/inference.py \
    --model_path ./merged_model.safetensors \
    --gemma_path $GEMMA_MODEL_PATH \
    --prompt "A man plays acoustic guitar on a wooden stage, warm applause from the audience" \
    --seed 42 \
    --output_dir ./results
```

<details>
<summary><strong>Arguments</strong></summary>

| Argument | Default | Description |
|---|---|---|
| `--model_path` | (required) | Path to merged `.safetensors` model |
| `--gemma_path` | env `GEMMA_MODEL_PATH` | Path to Gemma 3 text encoder |
| `--prompt` | (required) | Text prompt for generation |
| `--num_frames` | `121` | Number of video frames |
| `--height` / `--width` | model default | Video resolution |
| `--num_inference_steps` | model default | Number of denoising steps |
| `--video_guidance_scale` | model default | Video CFG scale |
| `--audio_guidance_scale` | model default | Audio CFG scale |
| `--seed` | `42` | Random seed |
| `--no_audio` | `false` | Disable audio generation |
| `--dtype` | `bf16` | Inference precision |

</details>

Outputs are saved to `--output_dir`: `.mp4` (video with audio) and `.wav` (audio only).

## 🖊️ Citation

```bibtex
@article{omninft2025,
  title={OmniNFT: Branch-Aware GRPO for Audio-Video Generation},
  year={2025}
}
```

## 🤝 Acknowledgements

[LTX-2](https://github.com/Lightricks/LTX-2) · [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)

---

## ⚠️ License

Research use only. See individual submodule licenses (HPSv3, ImageBind, LTX-Video, etc.) for their terms.