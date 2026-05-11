# AGENTS.md

This file provides guidance to AI coding assistants (Claude, Cursor, etc.) when working with code in this repository.

## Project Overview

**LTX Trainer** is a training toolkit for fine-tuning the Lightricks LTX audio-video generation models. It supports:

- **LoRA training** - Efficient fine-tuning with adapters
- **Full fine-tuning** - Complete model training
- **Audio-video training** - Joint audio and video generation
- **IC-LoRA training** - In-context control adapters for video-to-video transformations

**Supported model versions:**

- **LTX-2** (19B, initial audio-video model)
- **LTX-2.3** (22B, improved text conditioning and audio quality)

Version detection is fully automatic — ltx-core reads the checkpoint config and selects the correct architecture
components. The trainer does not need version-specific code paths.

**Key Dependencies:**

- **[`ltx-core`](../ltx-core/)** - Core model implementations (transformer, VAE, text encoder, scheduler)
- **[`ltx-pipelines`](../ltx-pipelines/)** - Inference pipeline components

> **Important:** This trainer only supports **LTX-2 and later** (audio-video models). The older LTXV (video-only) models
> are not supported.

## Architecture Overview

### Package Structure

```
packages/ltx-trainer/
├── src/ltx_trainer/              # Main training module
│   ├── __init__.py               # Logger setup, path config
│   ├── config.py                 # Pydantic configuration models
│   ├── config_display.py         # Config pretty-printing
│   ├── trainer.py                # Main training orchestration with Accelerate
│   ├── model_loader.py           # Model loading using ltx-core
│   ├── validation_sampler.py     # Inference for validation samples
│   ├── datasets.py               # PrecomputedDataset, DummyDataset
│   ├── training_strategies/      # Strategy pattern for different training modes
│   │   ├── __init__.py           # Factory function: get_training_strategy()
│   │   ├── base_strategy.py      # TrainingStrategy ABC, ModelInputs, TrainingStrategyConfigBase
│   │   ├── text_to_video.py      # TextToVideoStrategy, TextToVideoConfig
│   │   └── video_to_video.py     # VideoToVideoStrategy, VideoToVideoConfig
│   ├── timestep_samplers.py      # Flow matching timestep sampling
│   ├── gemma_8bit.py             # 8-bit Gemma text encoder loading (bitsandbytes)
│   ├── quantization.py           # Transformer INT8/INT4/FP8 quantization
│   ├── captioning.py             # Video captioning utilities
│   ├── video_utils.py            # Video I/O and processing
│   ├── gpu_utils.py              # GPU memory helpers
│   ├── hf_hub_utils.py           # HuggingFace Hub integration
│   ├── progress.py               # Training progress display
│   └── utils.py                  # Image I/O helpers
├── scripts/                      # User-facing CLI tools
│   ├── train.py                  # Main training script
│   ├── process_dataset.py        # Dataset preprocessing (latents + captions)
│   ├── process_videos.py         # Video latent encoding
│   ├── process_captions.py       # Text embedding computation
│   ├── caption_videos.py         # Automatic video captioning
│   ├── decode_latents.py         # Latent decoding for debugging
│   ├── inference.py              # Inference with trained models
│   ├── compute_reference.py      # Generate IC-LoRA reference videos
│   └── split_scenes.py           # Scene detection and splitting
├── configs/                      # Example training configurations
│   ├── ltx2_av_lora.yaml         # Audio-video LoRA training
│   ├── ltx2_av_lora_low_vram.yaml
│   ├── ltx2_v2v_ic_lora.yaml     # IC-LoRA video-to-video
│   └── accelerate/               # FSDP, DDP configs
├── tests/                        # Pytest tests
└── docs/                         # Documentation
```

### Key Architectural Patterns

**Model Loading:**

- `ltx_trainer.model_loader` provides component loaders using `ltx-core`
- Individual loaders: `load_transformer()`, `load_video_vae_encoder()`, `load_video_vae_decoder()`,
  `load_text_encoder()`, `load_embeddings_processor()`, etc.
- Combined loader: `load_model()` returns `LtxModelComponents` dataclass
- Uses `SingleGPUModelBuilder` from ltx-core internally
- Text encoder and embeddings processor are loaded separately (the text encoder only needs Gemma weights; the embeddings processor only needs the LTX checkpoint)
- 8-bit text encoder loading via `gemma_8bit.py` (bitsandbytes)

**Training Flow:**

1. Configuration loaded via Pydantic models in `config.py`
2. `LtxvTrainer` class orchestrates the training loop
3. Text encoder loaded on GPU → validation embeddings cached → heavy components unloaded (only `embeddings_processor`
   kept)
4. Each training step: embedding connectors applied → strategy prepares `ModelInputs` → transformer forward pass →
   strategy computes loss
5. Training strategies (`TextToVideoStrategy`, `VideoToVideoStrategy`) handle mode-specific logic
6. Accelerate handles distributed training, mixed precision, and device placement
7. Data flows as precomputed latents through `PrecomputedDataset`

**Model Interface (Modality-based):**

```python
from ltx_core.model.transformer.modality import Modality

video = Modality(
    enabled=True,
    latent=video_latents,  # [B, seq_len, 128] patchified latent tokens
    sigma=sigma,  # [B,] current noise level (per-batch)
    timesteps=video_timesteps,  # [B, seq_len] per-token timestep embeddings
    positions=video_positions,  # [B, 3, seq_len, 2] positional coordinates
    context=video_embeds,  # text conditioning embeddings
    context_mask=None,  # optional attention mask for text context
)
audio = Modality(
    enabled=True,
    latent=audio_latents,
    sigma=sigma,
    timesteps=audio_timesteps,
    positions=audio_positions,  # [B, 1, seq_len, 2]
    context=audio_embeds,
    context_mask=None,
)

# Forward pass returns predictions for both modalities
video_pred, audio_pred = model(video=video, audio=audio, perturbations=None)
```

> **Note:** `Modality` is immutable (frozen dataclass). Use `dataclasses.replace()` to modify.

**`sigma` vs `timesteps`:** These serve different roles. `timesteps` is per-token (e.g. `sigma * denoise_mask` —
conditioning tokens get 0, noisy tokens get sigma). `sigma` is per-batch and is used for prompt AdaLN conditioning (
LTX-2.3) and cross-modality (video↔audio) attention conditioning (both versions).

**Configuration System:**

- All config in `src/ltx_trainer/config.py`
- Main class: `LtxTrainerConfig`
- Training strategy configs: `TextToVideoConfig`, `VideoToVideoConfig`
- Uses Pydantic field validators and model validators
- Config uses `extra="forbid"` — unknown fields cause validation errors
- Config files in `configs/` directory

## LTX-2 vs LTX-2.3: Differences

Both model versions share the same latent space interface (see [Latent Space Constants](#latent-space-constants)).
The differences lie in how text conditioning and audio generation work. Version detection is automatic via checkpoint
config — the trainer uses a unified API.

| Component             | LTX-2 (19B)                                                                     | LTX-2.3 (22B)                                                                                       |
|-----------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Feature extractor     | `FeatureExtractorV1`: single `aggregate_embed`, same output for video and audio | `FeatureExtractorV2`: separate `video_aggregate_embed` + `audio_aggregate_embed`, per-token RMSNorm |
| Caption projection    | Inside the transformer (`caption_projection`)                                   | Inside the feature extractor (before connector)                                                     |
| Embeddings connectors | Same dimensions for video and audio                                             | Separate dimensions (`AudioEmbeddings1DConnectorConfigurator`)                                      |
| Prompt AdaLN          | Not present (`cross_attention_adaln=False`)                                     | Active — modulates cross-attention to text using `sigma`                                            |
| Vocoder               | HiFi-GAN (`Vocoder`)                                                            | BigVGAN v2 + bandwidth extension (`VocoderWithBWE`)                                                 |

**How version detection works in ltx-core:**

- **Feature extractor:** `_create_feature_extractor()` checks for V2 config keys (`caption_proj_before_connector`,
  etc.). Present → V2; absent → V1.
- **Vocoder:** `VocoderConfigurator` checks for `config["vocoder"]["bwe"]`. Present → `VocoderWithBWE`; absent →
  `Vocoder`.
- **Transformer:** `_build_caption_projections()` checks `caption_proj_before_connector`. True (V2) → no caption
  projection in transformer; False (V1) → caption projection created in transformer.
- **Embeddings connectors:** `AudioEmbeddings1DConnectorConfigurator` reads `audio_connector_*` keys, falling back to
  video connector keys for V1 backward compatibility.

## Text Encoder Pipeline

The `GemmaTextEncoder` implements a 3-block pipeline:

1. **Block 1 — Gemma LLM:** Tokenizes text → runs through Gemma → extracts hidden states
2. **Block 2 — Feature extractor:** Hidden states → normalized features (V1: single stream duplicated for video/audio;
   V2: separate video and audio projections)
3. **Block 3 — Embeddings processor:** Features → embeddings connectors → final context embeddings for the transformer

**Precomputed embeddings (offline):** `process_captions.py` runs Blocks 1+2 via `text_encoder.precompute()` and saves
the results. Block 3 (connectors) is applied during training via
`text_encoder.embeddings_processor.create_embeddings()`.

**Precomputed embeddings formats:**

- **New format** (from `precompute()`): saves `video_prompt_embeds`, `audio_prompt_embeds` (optional),
  `prompt_attention_mask`
- **Legacy format** (from old `_preprocess_text()`): saves `prompt_embeds`, `prompt_attention_mask`

The trainer handles both formats in `_training_step()`: if `video_prompt_embeds` is present, it uses the new format;
otherwise, it duplicates `prompt_embeds` for both modalities (mirroring V1 behavior).

**After caching validation embeddings**, the trainer unloads heavy components to free VRAM:

```python
self._text_encoder.model = None
self._text_encoder.tokenizer = None
self._text_encoder.feature_extractor = None
# Only embeddings_processor (connectors) remains — used during training
```

## Latent Space Constants

These values are shared across all supported model versions:

| Constant                     | Value                            | Where used                                                |
|------------------------------|----------------------------------|-----------------------------------------------------------|
| Video latent channels        | 128                              | VAE encoder/decoder, patchifier, `VideoLatentShape`       |
| Spatial compression          | 32× (H and W)                    | `SpatioTemporalScaleFactors.default()`, config validators |
| Temporal compression         | 8×                               | `SpatioTemporalScaleFactors.default()`, config validators |
| Frame constraint             | `frames % 8 == 1`                | Config validators, validation sampler                     |
| Resolution constraint        | Width and height divisible by 32 | Config validators, validation sampler                     |
| Audio latent channels        | 8                                | `AudioLatentShape`, audio patchifier                      |
| Audio mel bins               | 16                               | `AudioLatentShape`, audio patchifier                      |
| Patchified token dim (video) | 128 (`128 × 1 × 1 × 1`)          | Transformer `in_channels`                                 |
| Patchified token dim (audio) | 128 (`8 × 16`)                   | Transformer `audio_in_channels`                           |

## Development Commands

### Setup and Installation

```bash
# From the repository root
uv sync
cd packages/ltx-trainer
```

### Code Quality

```bash
# Run ruff linting and formatting
uv run ruff check .
uv run ruff format .

# Run pre-commit checks
uv run pre-commit run --all-files
```

### Running Tests

```bash
cd packages/ltx-trainer
uv run pytest
```

### Running Training

```bash
# Single GPU
uv run python scripts/train.py configs/ltx2_av_lora.yaml

# Multi-GPU with Accelerate
uv run accelerate launch scripts/train.py configs/ltx2_av_lora.yaml
```

## Code Standards

### Type Hints

- **Always use type hints** for all function arguments and return values
- Use Python 3.10+ syntax: `list[str]` not `List[str]`, `str | Path` not `Union[str, Path]`
- Use `pathlib.Path` for file operations

### Class Methods

- Mark methods as `@staticmethod` if they don't access instance or class state
- Use `@classmethod` for alternative constructors

### AI/ML Specific

- Use `@torch.inference_mode()` for inference (prefer over `@torch.no_grad()`)
- Use `accelerator.device` for distributed compatibility
- Support mixed precision (`bfloat16` via dtype parameters)
- Use gradient checkpointing for memory-intensive training

### Logging

- Use `from ltx_trainer import logger` for all messages
- Avoid print statements in production code

## Important Files & Modules

### Configuration (CRITICAL)

**`src/ltx_trainer/config.py`** - Master config definitions

Key classes:

- `LtxTrainerConfig` - Main configuration container
- `ModelConfig` - Model paths, training mode (`lora` | `full`), checkpoint loading
- `TrainingStrategyConfig` - Union of `TextToVideoConfig` | `VideoToVideoConfig` (discriminated by `name`)
- `LoraConfig` - Rank, alpha, dropout, target modules
- `OptimizationConfig` - Learning rate, batch size, gradient accumulation, scheduler, gradient checkpointing
- `AccelerationConfig` - Mixed precision, quantization, 8-bit text encoder
- `DataConfig` - Preprocessed data root, dataloader workers
- `ValidationConfig` - Prompts, video dimensions, CFG/STG guidance, audio generation, inference steps
- `CheckpointsConfig` - Save interval, retention, precision
- `FlowMatchingConfig` - Timestep sampling mode and parameters
- `HubConfig` - HuggingFace Hub push settings
- `WandbConfig` - Weights & Biases logging

**⚠️ When modifying config.py:**

1. Update ALL config files in `configs/`
2. Update `docs/configuration-reference.md`
3. Test that all configs remain valid

### Training Core

**`src/ltx_trainer/trainer.py`** - Main training loop (`LtxvTrainer`)

- Implements distributed training with Accelerate
- Handles mixed precision, gradient accumulation, checkpointing
- `_training_step()` applies embedding connectors then delegates to strategy
- `_load_text_encoder_and_cache_embeddings()` loads the text encoder + embeddings processor, caches validation embeddings, then unloads the Gemma LLM (keeps only the embeddings processor connectors for training)
- Uses training strategies for mode-specific logic

**`src/ltx_trainer/training_strategies/`** - Strategy pattern

- `base_strategy.py`: `TrainingStrategy` ABC, `ModelInputs` dataclass
- `text_to_video.py`: Standard text-to-video (with optional audio)
- `video_to_video.py`: IC-LoRA video-to-video transformations

Key methods each strategy implements:

- `get_data_sources()` - Required data directories
- `prepare_training_inputs()` - Convert batch to `ModelInputs` with `Modality` objects
- `compute_loss()` - Calculate training loss (velocity prediction, MSE with masking)
- `requires_audio` property - Whether audio components needed

**`src/ltx_trainer/model_loader.py`** - Model loading

Component loaders:

- `load_transformer()` → `LTXModel`
- `load_video_vae_encoder()` → `VideoEncoder`
- `load_video_vae_decoder()` → `VideoDecoder`
- `load_audio_vae_decoder()` → `AudioDecoder`
- `load_vocoder()` → `Vocoder` or `VocoderWithBWE` (auto-detected)
- `load_text_encoder(gemma_model_path)` → `GemmaTextEncoder` (pure Gemma LLM, no checkpoint needed)
- `load_embeddings_processor(checkpoint_path)` → `EmbeddingsProcessor` (feature extractor + connectors)
- `load_model()` → `LtxModelComponents` (convenience wrapper)

**`src/ltx_trainer/validation_sampler.py`** - Inference for validation

Uses ltx-core components for denoising:

- `LTX2Scheduler` for sigma scheduling
- `EulerDiffusionStep` for diffusion steps
- `CFGGuider` for classifier-free guidance
- `STGGuider` for spatio-temporal guidance

**`src/ltx_trainer/timestep_samplers.py`** - Flow matching timestep sampling

- `UniformTimestepSampler` - Uniform sampling in `[min, max]`
- `ShiftedLogitNormalTimestepSampler` - Stretched shifted logit-normal distribution with:
    - Shift determined by sequence length (more noise at higher token counts)
    - Percentile stretching for better `[0, 1]` coverage
    - Uniform fallback (10% of samples) to prevent distribution collapse
    - Reflection around `eps` for numerical stability near zero

**`src/ltx_trainer/gemma_8bit.py`** - 8-bit text encoder loading

Bypasses ltx-core's standard loading path to enable bitsandbytes 8-bit quantization of the Gemma backbone. Manually
constructs the `GemmaTextEncoder` with quantized model, feature extractor, and embeddings processor.

### Data

**`src/ltx_trainer/datasets.py`** - Dataset handling

- `PrecomputedDataset` loads pre-computed VAE latents and text embeddings
- Supports video latents, audio latents, text embeddings, reference latents (for IC-LoRA)
- Handles legacy patchified format `[seq_len, C]` → automatically unpatchifies to `[C, F, H, W]`
- `DummyDataset` for benchmarking and minimal testing

## Common Development Tasks

### Adding a New Configuration Parameter

1. Add field to appropriate config class in `src/ltx_trainer/config.py`
2. Add validator if needed
3. Update ALL config files in `configs/`
4. Update `docs/configuration-reference.md`

### Implementing a New Training Strategy

1. Create new file in `src/ltx_trainer/training_strategies/`
2. Create config class inheriting `TrainingStrategyConfigBase`
3. Create strategy class inheriting `TrainingStrategy`
4. Implement: `get_data_sources()`, `prepare_training_inputs()`, `compute_loss()`
5. Add to `__init__.py`: import, add to `TrainingStrategyConfig` union, update factory
6. Add discriminator tag to config.py's `TrainingStrategyConfig`
7. Create example config file in `configs/`

### Working with Modalities

```python
from dataclasses import replace
from ltx_core.model.transformer.modality import Modality

# Create modality — all fields except enabled and masks are required
video = Modality(
    enabled=True,
    latent=latents,  # [B, seq_len, 128]
    sigma=sigma,  # [B,] — the per-batch noise level
    timesteps=timesteps,  # [B, seq_len] — per-token (sigma * denoise_mask)
    positions=positions,  # [B, 3, seq_len, 2]
    context=context,  # text embeddings from embeddings_processor
    context_mask=None,
)

# Update (immutable — must use replace)
video = replace(video, latent=new_latent, sigma=new_sigma, timesteps=new_timesteps)

# Disable a modality
audio = replace(audio, enabled=False)
```

### Working with the Text Encoder

```python
# Full forward pass (used for validation — runs all 3 blocks)
video_embeds, audio_embeds, attention_mask = text_encoder(prompt)

# Precompute features (used in process_captions.py — runs blocks 1+2 only)
video_features, audio_features, attention_mask = text_encoder.precompute(prompt, padding_side="left")

# Apply connectors during training (block 3 only)
additive_mask = text_encoder._convert_to_additive_mask(attention_mask, video_features.dtype)
video_embeds, audio_embeds, binary_mask = text_encoder.embeddings_processor.create_embeddings(
    video_features, audio_features, additive_mask
)
```

## Debugging Tips

**Training Issues:**

- Check logs first (rich logger provides context)
- GPU memory: Look for OOM errors, enable `enable_gradient_checkpointing: true`
- Distributed training: Check `accelerator.state` and device placement

**Model Loading:**

- Ensure `model_path` points to a local `.safetensors` file
- Ensure `text_encoder_path` points to a Gemma model directory
- URLs are NOT supported for model paths
- For 8-bit loading: ensure `bitsandbytes` is installed

**Configuration:**

- Validation errors: Check validators in `config.py`
- Unknown fields: Config uses `extra="forbid"` — all fields must be defined
- Strategy validation: IC-LoRA requires `reference_videos` in validation config
- Video-to-video strategy requires `training_mode: "lora"`

**Precomputed Data:**

- Legacy data (`prompt_embeds`) works via backward-compat in `_training_step()`
- New data (`video_prompt_embeds` + `audio_prompt_embeds`) is the expected format
- Latents must be in `[C, F, H, W]` format (legacy `[seq_len, C]` is auto-converted)

## Key Constraints

### Frame Requirements

Frames must satisfy `frames % 8 == 1`:

- ✅ Valid: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121
- ❌ Invalid: 24, 32, 48, 64, 100

### Resolution Requirements

Width and height must be divisible by 32.

### Model Paths

- Must be local paths (URLs not supported)
- `model_path`: Path to `.safetensors` checkpoint
- `text_encoder_path`: Path to Gemma model directory

### Platform Requirements

- Linux required (uses `triton` which is Linux-only)
- CUDA GPU with 24GB+ VRAM recommended (80GB+ for full fine-tuning)

## Reference: ltx-core Key Components

```
packages/ltx-core/src/ltx_core/
├── model/
│   ├── transformer/
│   │   ├── model.py                # LTXModel (diffusion transformer)
│   │   ├── modality.py             # Modality dataclass
│   │   ├── transformer.py          # BasicAVTransformerBlock
│   │   ├── transformer_args.py     # TransformerArgsPreprocessor (sigma → prompt AdaLN)
│   │   ├── model_configurator.py   # LTXModelConfigurator (version-aware)
│   │   └── timestep_embedding.py   # Timestep/sigma embedding
│   ├── video_vae/
│   │   ├── video_vae.py            # VideoEncoder, VideoDecoder
│   │   └── model_configurator.py   # VideoEncoderConfigurator, VideoDecoderConfigurator
│   ├── audio_vae/
│   │   ├── audio_vae.py            # AudioEncoder, AudioDecoder
│   │   └── vocoder.py              # Vocoder, VocoderWithBWE (output_sampling_rate)
│   └── common/                     # Shared model components
├── text_encoders/gemma/
│   ├── __init__.py                 # Exports: GemmaTextEncoder, GemmaTextEncoderConfigurator,
│   │                               #   AV_GEMMA_TEXT_ENCODER_KEY_OPS, GEMMA_MODEL_OPS,
│   │                               #   module_ops_from_gemma_root
│   ├── encoders/
│   │   ├── base_encoder.py         # GemmaTextEncoder (unified 3-block pipeline)
│   │   └── encoder_configurator.py # GemmaTextEncoderConfigurator, _create_feature_extractor
│   ├── feature_extractor.py        # FeatureExtractorV1 (19B), FeatureExtractorV2 (22B)
│   ├── embeddings_connector.py     # Embeddings1DConnector, Embeddings1DConnectorConfigurator,
│   │                               #   AudioEmbeddings1DConnectorConfigurator
│   ├── embeddings_processor.py     # EmbeddingsProcessor (wraps video + audio connectors)
│   └── tokenizer.py               # LTXVGemmaTokenizer
├── components/
│   ├── schedulers.py               # LTX2Scheduler
│   ├── diffusion_steps.py          # EulerDiffusionStep
│   ├── guiders.py                  # CFGGuider, STGGuider
│   └── patchifiers.py              # VideoLatentPatchifier, AudioPatchifier
├── conditioning/                   # ConditioningItem, mask_utils, types
├── tools.py                        # VideoLatentTools, AudioLatentTools
├── loader/
│   ├── single_gpu_model_builder.py # SingleGPUModelBuilder
│   ├── sft_loader.py              # SafetensorsModelStateDictLoader
│   └── sd_ops.py                  # Key remapping (SDOps)
└── types.py                       # SpatioTemporalScaleFactors, VideoLatentShape, AudioLatentShape
```
