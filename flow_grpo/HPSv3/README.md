<div align="center">

# HPSv3: Towards Wide-Spectrum Human Preference Score (ICCV 2025)

[![Project Website](https://img.shields.io/badge/🌐-Project%20Website-deepgray)](https://mizzenai.github.io/HPSv3.project/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.03789-b31b1b.svg)](https://arxiv.org/abs/2508.03789)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue.svg)](https://arxiv.org/abs/2508.03789)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/MizzenAI/HPSv3)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-green)](https://huggingface.co/datasets/MizzenAI/HPDv3)
[![PyPI](https://img.shields.io/pypi/v/hpsv3)](https://pypi.org/project/hpsv3/)

**Yuhang Ma**<sup>1,3*</sup>&ensp; **Yunhao Shui**<sup>1,4*</sup>&ensp; **Xiaoshi Wu**<sup>2</sup>&ensp; **Keqiang Sun**<sup>1,2†</sup>&ensp; **Hongsheng Li**<sup>2,5,6†</sup>

<sup>1</sup>Mizzen AI&ensp;&ensp; <sup>2</sup>CUHK MMLab&ensp;&ensp; <sup>3</sup>King’s College London&ensp;&ensp; <sup>4</sup>Shanghai Jiaotong University&ensp;&ensp; 

<sup>5</sup>Shanghai AI Laboratory&ensp;&ensp; <sup>6</sup>CPII, InnoHK&ensp;&ensp; 

<sup>*</sup>Equal Contribution&ensp; <sup>†</sup>Equal Advising

</div>


## 📖 Introduction

This is the official implementation for the paper: [HPSv3: Towards Wide-Spectrum Human Preference Score](https://arxiv.org/abs/2508.03789).
First, we introduce a VLM-based preference model **HPSv3**, trained on a "wide spectrum" preference dataset **HPDv3** with 1.08M text-image pairs and 1.17M annotated pairwise comparisons, covering both state-of-the-art and earlier generative models, as well as high- and low-quality real-world images. Second, we propose a novel reasoning approach for iterative image refinement, **CoHP(Chain-of-Human-Preference)**, which efficiently improves image quality without requiring additional training data.

<p align="center">
  <img src="assets/teaser.png" alt="Teaser" width="900"/>
</p>


## ✨ Updates
- **[2025-08-19]** 🖼️ We release [DanceGRPO](https://github.com/XueZeyue/DanceGRPO) results of HPSv3! Great thanks to [XueZeyue](https://github.com/XueZeyue) for training it!
- **[2025-08-08]** 🎉 We release [HPDv3](https://huggingface.co/datasets/MizzenAI/HPDv3) dataset!.
- **[2025-08-06]** 🎉 We release HPSv3: inference code, training code, cohp code and [HPSv3 model weights](https://huggingface.co/MizzenAI/HPSv3). And [PyPI Package](https://pypi.org/project/hpsv3/).

## 📑 Table of Contents
1. [🚀 Quick Start](#🚀-quick-start)
2. [🌐 Gradio Demo](#🌐-gradio-demo)
3. [🏋️ Training](#🏋️-training)
4. [📊 Benchmark](#📊-benchmark)
5. [🎯 CoHP (Chain-of-Human-Preference)](#🎯-cohp-chain-of-human-preference)

---

## 🚀 Quick Start

HPSv3 is a state-of-the-art human preference score model for evaluating image quality and prompt alignment. It builds upon the Qwen2-VL architecture to provide accurate assessments of generated images.

### 💻 Installation

<!-- # Method 1: Pypi download and install for inference.
pip install hpsv3 -->

```bash
# Method 1: Pypi download and install for inference.
pip install hpsv3

# Method 2: Install locally for development or training.
git clone https://github.com/MizzenAI/HPSv3.git
cd HPSv3

conda env create -f environment.yaml
conda activate hpsv3
# Recommend: Install flash-attn
pip install flash-attn==2.7.4.post1

pip install -e .
```

### 🛠️ Basic Usage

#### Simple Inference Example

```python
from hpsv3 import HPSv3RewardInferencer

# Initialize the model
inferencer = HPSv3RewardInferencer(device='cuda')

# Evaluate images
image_paths = ["assets/example1.png", "assets/example2.png"]
prompts = [
  "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker",
  "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker"
]

# Get preference scores
rewards = inferencer.reward(prompts, image_paths=image_paths)
scores = [reward[0].item() for reward in rewards]  # Extract mu values
print(f"Image scores: {scores}")
```

---

## 🌐 Gradio Demo

Launch an interactive web interface to test HPSv3:

```bash
python gradio_demo/demo.py
```

The demo will be available at `http://localhost:7860` and provides:

<p align="left">
  <img src="assets/gradio.png" alt="Gradio Demo" width="500"/>
</p>



## 📁 Dataset

### Human Preference Dataset v3

Human Preference Dataset v3 (HPD v3) comprises 1.08M text-image pairs and 1.17M annotated pairwise data. To modeling the wide spectrum of human preference, we introduce newest state-of-the-art generative models and high quality real photographs while maintaining old models and lower quality real images.
<p align="left">
  <img src="assets/datasetvisual_0.jpg" alt="dataset" width="500"/>
</p>
<details close>
<summary>Detail information of HPD v3</summary>

| Image Source | Type | Num Image | Prompt Source | Split |
|--------------|------|-----------|---------------|-------|
| High Quality Image (HQI) | Real Image | 57759 | VLM Caption | Train & Test |
| MidJourney | - | 331955 | User | Train |
| CogView4 | DiT | 400 | HQI+HPDv2+JourneyDB | Test |
| FLUX.1 dev | DiT | 48927 | HQI+HPDv2+JourneyDB | Train & Test |
| Infinity | Autoregressive | 27061 | HQI+HPDv2+JourneyDB | Train & Test |
| Kolors | DiT | 49705 | HQI+HPDv2+JourneyDB | Train & Test |
| HunyuanDiT | DiT | 46133 | HQI+HPDv2+JourneyDB | Train & Test |
| Stable Diffusion 3 Medium | DiT | 49266 | HQI+HPDv2+JourneyDB | Train & Test |
| Stable Diffusion XL | Diffusion | 49025 | HQI+HPDv2+JourneyDB | Train & Test |
| Pixart Sigma | Diffusion | 400 | HQI+HPDv2+JourneyDB | Test |
| Stable Diffusion 2 | Diffusion | 19124 | HQI+JourneyDB | Train & Test |
| CogView2 | Autoregressive | 3823 | HQI+JourneyDB | Train & Test |
| FuseDream | Diffusion | 468 | HQI+JourneyDB | Train & Test |
| VQ-Diffusion | Diffusion | 18837 | HQI+JourneyDB | Train & Test |
| Glide | Diffusion | 19989 | HQI+JourneyDB | Train & Test |
| Stable Diffusion 1.4 | Diffusion | 18596 | HQI+JourneyDB | Train & Test |
| Stable Diffusion 1.1 | Diffusion | 19043 | HQI+JourneyDB | Train & Test |
| Curated HPDv2 | - | 327763 | - | Train |
</details>

### Download HPDv3
<!-- ```
HPDv3 is comming soon! Stay tuned!
``` -->
```bash
huggingface-cli download --repo-type dataset MizzenAI/HPDv3 --local-dir /your-local-dataset-path
```

### Pairwise Training Data Format

**Important Note: For simplicity, path1's image is always the prefered one**

#### All Annotated Pairs (`all.json`)

**Important Notes: In HPDv3, we simply put the preferred sample at the first place (path1)**

`all.json` contains **all** annotated pairs except for test.

```bash
[
    # samples from HPDv3 annotation pipeline 
    {
    "prompt": "Description of the visual content or the generation prompt.",
    "choice_dist": [12, 7],           # Distribution of votes from annotators (12 votes for image1, 7 votes for image2)
    "confidence": 0.9999907,         # Confidence score reflecting preference reliability, based on annotators' capabilities (independent of choice_dist)
    "path1": "images/uuid1.jpg",     # File path to the preferred image
    "path2": "images/uuid2.jpg",     # File path to the non-preferred image
    "model1": "flux",                # Model used to generate the preferred image (path1)
    "model2": "infinity"             # Model used to generate the non-preferred image (path2)
    },
    # samples from Midjourney
    {
    "prompt": "Description of the visual content or the generation prompt.",
    "choice_dist": null,             # No distribution of votes Information from Discord
    "confidence": null,              # No Confidence Information from Discord
    "path1": "images/uuid1.jpg",     # File path to the preferred image.
    "path2": "images/uuid2.jpg",     # File path to the non-preferred image.
    "model1": "midjourney",          # Comparsion between images generated from midjourney 
    "model2": "midjourney"           # Comparsion between images generated from midjourney 
    },
    # samples from Curated HPDv2
    {
    "prompt": "Description of the visual content or the generation prompt.",
    "choice_dist": null,              # No distribution of votes Information from the original HPDv2 traindataset
    "confidence": null,               # No Confidence Information from the original HPDv2 traindataset
    "path1": "images/uuid1.jpg",     # File path to the preferred image.
    "path2": "images/uuid2.jpg",     # File path to the non-preferred image.
    "model1": "hpdv2",          # No specific model name in the original HPDv2 traindataset, set to hpdv2 
    "model2": "hpdv2"           # No specific model name in the original HPDv2 traindataset, set to hpdv2 
    },
]
```

#### Train set (`train.json`)
We sample part of training data from `all.json` to build training dataset `train.json`. Moreover, to improve robustness, we integrate random sampled part of data from [Pick-a-pic](https://huggingface.co/datasets/pickapic-anonymous/pickapic_v1) and [ImageRewardDB](https://huggingface.co/datasets/zai-org/ImageRewardDB), which is `pickapic.json` and `imagereward.json`. For these two datasets, we only provide the pair infomation, and its corresponding image can be found in their official dataset repository.


#### Test Set (`test.json`)
```bash
[
    {
        "prompt": "Description of the visual content",
        "path1": "images/uuid1.jpg",     # Preferred sample
        "path2": "images/uuid2.jpg",     # Unpreferred sample
        "model1": "flux",                # Model used to generate the preferred sample (path1).
        "model2": "infinity",            # Model used to generate the non-preferred sample (path2).

    }
]
```

## 🏋️ Training

### 🚀 Training Command

```bash
# Use Method 2 to install locally
git clone https://github.com/MizzenAI/HPSv3.git
cd HPSv3

conda env create -f environment.yaml
conda activate hpsv3
# Recommend: Install flash-attn
pip install flash-attn==2.7.4.post1

pip install -e .

# Train with 7B model
deepspeed hpsv3/train.py --config hpsv3/config/HPSv3_7B.yaml
```

<details close>
<summary>Important Config Argument</summary>

| Configuration Section | Parameter | Value | Description |
|----------------------|-----------|-------|-------------|
| **Model Configuration** | `rm_head_type` | `"ranknet"` | Type of reward model head architecture |
| | `lora_enable` | `False` | Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning. If `False`, language tower is fully trainable|
| | `vision_lora` | `False` | Apply LoRA specifically to vision components. If `False`, vision tower is fully trainable|
| | `model_name_or_path` | `"Qwen/Qwen2-VL-7B-Instruct"` | Path to the base model checkpoint |
| **Data Configuration** | `confidence_threshold` | `0.95` | Minimum confidence score for training data |
| | `train_json_list` | `[example_train.json]` | List of training data files |
| | `test_json_list` | `[validation_sets]` | List of validation datasets with names |
| | `output_dim` | `2` | Output dimension of the reward head for $\mu$ and $\sigma$|
| | `loss_type` | `"uncertainty"` | Loss function type for training |
</details>

---

## 📊 Benchmark
To evaluate **HPSv3 preference accuracy** or **human preference score of image generation model**, follow the detail instruction is in [Evaluate Insctruction](evaluate/README.md)

<details open>
<summary> Preference Accuracy of HPSv3 </summary>

| Model | ImageReward | Pickscore | HPDv2 | HPDv3 |
|------|-------------|-----------|-------|-------|
| [CLIP ViT-H/14](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | 57.1 | 60.8 | 65.1 | 48.6 |
| [Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) | 57.4 | 56.8 | 76.8 | 59.9 |
| [ImageReward](https://github.com/THUDM/ImageReward) | 65.1 | 61.1 | 74.0 | 58.6 |
| [PickScore](https://github.com/yuvalkirstain/PickScore) | 61.6 | <u>70.5</u> | 79.8 | <u>65.6</u> |
| [HPS](https://github.com/tgxs002/align_sd) | 61.2 | 66.7 | 77.6 | 63.8 |
| [HPSv2](https://github.com/tgxs002/HPSv2) | 65.7 | 63.8 | 83.3 | 65.3 |
| [MPS](https://github.com/Kwai-Kolors/MPS) | **67.5** | 63.1 | <u>83.5</u> | 64.3 |
| HPSv3 | <u>66.8</u> | **72.8** | **85.4** | **76.9** |

</details>

<details open>
<summary> Image Generation Benchmark of HPSv3 </summary>

| Model | Overall | Characters | Arts | Design | Architecture | Animals | Natural Scenery | Transportation | Products | Others | Plants | Food | Science |
|------|---------|------------|------|--------|--------------|---------|-----------------|----------------|----------|--------|--------|------|---------|
| **Kolors** | **10.55** | **11.79** | **10.47** | **9.87** | <u>10.82</u> | **10.60** | 9.89 | <u>10.68</u> | <u>10.93</u> | **10.50** | **10.63** | <u>11.06</u> | <u>9.51</u> |
| **Flux-dev** | <u>10.43</u> | <u>11.70</u> | <u>10.32</u> | 9.39 | **10.93** | <u>10.38</u> | <u>10.01</u> | **10.84** | **11.24** | <u>10.21</u> | 10.38 | **11.24** | 9.16 |
| **Playgroundv2.5** | 10.27 | 11.07 | 9.84 | <u>9.64</u> | 10.45 | <u>10.38</u> | 9.94 | 10.51 | <u>10.62</u> | 10.15 | <u>10.62</u> | 10.84 | 9.39 |
| **Infinity** | 10.26 | 11.17 | 9.95 | 9.43 | 10.36 | 9.27 | **10.11** | 10.36 | 10.59 | 10.08 | 10.30 | 10.59 | **9.62** |
| **CogView4** | 9.61 | 10.72 | 9.86 | 9.33 | 9.88 | 9.16 | 9.45 | 9.69 | 9.86 | 9.45 | 9.49 | 10.16 | 8.97 |
| **PixArt-Σ** | 9.37 | 10.08 | 9.07 | 8.41 | 9.83 | 8.86 | 8.87 | 9.44 | 9.57 | 9.52 | 9.73 | 10.35 | 8.58 |
| **Gemini 2.0 Flash** | 9.21 | 9.98 | 8.44 | 7.64 | 10.11 | 9.42 | 9.01 | 9.74 | 9.64 | 9.55 | 10.16 | 7.61 | 9.23 |
| **SDXL** | 8.20 | 8.67 | 7.63 | 7.53 | 8.57 | 8.18 | 7.76 | 8.65 | 8.85 | 8.32 | 8.43 | 8.78 | 7.29 |
| **HunyuanDiT** | 8.19 | 7.96 | 8.11 | 8.28 | 8.71 | 7.24 | 7.86 | 8.33 | 8.55 | 8.28 | 8.31 | 8.48 | 8.20 |
| **Stable Diffusion 3 Medium** | 5.31 | 6.70 | 5.98 | 5.15 | 5.25 | 4.09 | 5.24 | 4.25 | 5.71 | 5.84 | 6.01 | 5.71 | 4.58 |
| **SD2** | -0.24 | -0.34 | -0.56 | -1.35 | -0.24 | -0.54 | -0.32 | 1.00 | 1.11 | -0.01 | -0.38 | -0.38 | -0.84 |

</details>

---

## 🎯 CoHP (Chain-of-Human-Preference)

COHP is our novel reasoning approach for iterative image refinement that efficiently improves image quality without requiring additional training data. It works by generating images with multiple diffusion models, selecting the best one using reward models, and then iteratively refining it through image-to-image generation.

<p align="left">
  <img src="assets/cohp.png" alt="cohp" width="600"/>
</p>

### 🚀 Usage

#### Basic Command

```bash
python hpsv3/cohp/run_cohp.py \
    --prompt "A beautiful sunset over mountains" \
    --index "sample_001" \
    --device "cuda:0" \
    --reward_model "hpsv3"
```

#### Parameters

- `--prompt`: Text prompt for image generation (required)
- `--index`: Unique identifier for saving results (required)  
- `--device`: GPU device to use (default: 'cuda:1')
- `--reward_model`: Reward model for scoring images
  - `hpsv3`: HPSv3 model (default, recommended)
  - `hpsv2`: HPSv2 model
  - `imagereward`: ImageReward model
  - `pickscore`: PickScore model

#### Supported Generation Models

COHP uses multiple state-of-the-art diffusion models for initial generation: **FLUX.1 dev**, **Kolors**, **Stable Diffusion 3 Medium**, **Playground v2.5**

#### How COHP Works

1. **Multi-Model Generation**: Generates images using all supported models
2. **Reward Scoring**: Evaluates each image using the specified reward model
3. **Best Model Selection**: Chooses the model that achieves the highest average score for its generated images
4. **Iterative Refinement**: Performs 4 rounds of image-to-image generation to improve quality
5. **Adaptive Strength**: Uses strength=0.8 for rounds 1-2, then 0.5 for rounds 3-4

---

## 🦾 Results as Reward Model

We perform [DanceGRPO](https://github.com/XueZeyue/DanceGRPO) as the reinforcement learning method. Here are some results.
All experiments using the same setting and we use **Stable Diffusion 1.4** as our backbone.

<p align="left">
  <img src="assets/rl1.jpg"  width="600"/>
</p>

<p align="left">
  <img src="assets/rl2.jpg"  width="600"/>
</p>


### More Results of HPsv3 as Reward Model (Stable Diffusion 1.4)
<p align="left">
  <img src="assets/rl_teaser.jpg" alt="cohp" width="600"/>
</p>

### Results of HPsv3 as Reward Model (FLUX.1 dev)
<p align="left">
  <img src="assets/rl3.jpg" alt="cohp" width="600"/>
</p>

## 📚 Citation

If you find HPSv3 useful in your research, please cite our work:

```bibtex
@misc{ma2025hpsv3widespectrumhumanpreference,
      title={HPSv3: Towards Wide-Spectrum Human Preference Score}, 
      author={Yuhang Ma and Xiaoshi Wu and Keqiang Sun and Hongsheng Li},
      year={2025},
      eprint={2508.03789},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.03789}, 
}
```


---

## 🙏 Acknowledgements

We would like to thank the [VideoAlign](https://github.com/KwaiVGI/VideoAlign) codebase for providing valuable references.

---
## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MizzenAI/HPSv3&type=Date)](https://www.star-history.com/#MizzenAI/HPSv3&Date)

## 💬 Support

For questions and support:
- **Issues**: [GitHub Issues](https://github.com/MizzenAI/HPSv3/issues)
- **Email**: yhshui@mizzen.ai & yhma@mizzen.ai
