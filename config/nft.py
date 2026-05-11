import importlib.util
import os
import sys
import ml_collections

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. 获取文件路径
file_path = os.path.join(os.path.dirname(__file__), "base.py")
module_name = "base"
# 2. 使用 importlib 动态加载
# 创建模块的规格 (Spec)
spec = importlib.util.spec_from_file_location(module_name, file_path)
# 根据规格创建模块对象
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.modules[module_name] = base


def get_config(name):
    return globals()[name]()


def _get_config(base_model="javis", n_gpus=1, bsz=4, gradient_step_per_epoch=1, dataset="audio_video", reward_fn={}, name="", prefix=""):
    config = base.get_config()

    config.mixed_precision = "bf16"
    config.activation_checkpointing = True

    config.base_model = base_model
    # config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    config.train_dataset = os.path.join(_PROJECT_ROOT, "dataset/vggsound/train_metadata_20k.jsonl")
    config.test_dataset = os.path.join(_PROJECT_ROOT, "dataset/vggsound/test_metadata_arena.jsonl")

    config.reward_route = ml_collections.ConfigDict()
    config.reward_route.video_keys = ["hpsv3_score_video", "videoalign_score"]
    config.reward_route.audio_keys = ["audiobox_aesthetics_score", "clap_score"]
    config.reward_route.sync_keys = ["av_align_score", "av_desync_reward", "av_desync_reward_remote"]
    
    config.sample.num_steps = 20
    config.train.timestep_fraction = 0.4

    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.noise_level = 0.7

    config.resolution_heigh = 512
    config.resolution_width = 768
    
    bsz = bsz
    config.sample.num_image_per_prompt = 8
    num_groups = (n_gpus * bsz) // config.sample.num_image_per_prompt

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = 14 if dataset == "geneval" else 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "audio_video_prompt"

    config.run_name = f"{prefix}/logs/nft_{base_model}_{name}"
    config.save_dir = f"{prefix}/logs/nft/{base_model}/{name}"
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.train.beta = 0.0001
    config.beta = 1.0
    config.train.adv_mode = "all"

    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config

target_modules = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v",
    "audio_to_video_attn.to_out.0",
    "video_to_audio_attn.to_q",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "video_to_audio_attn.to_out.0",
]


def exp0502_ltx_mllm_gdpo():
    reward_fn = {
        "videoalign_score": 1.0,
        "hpsv3_score_video": 1.5,
        "audiobox_aesthetics_score": 0.5,
        "clap_score": 1.0,
        "av_desync_reward": 1.0
    }

    exp_name = "exp0502_ltx_mllm_gdpo"
    prefix = os.environ.get("OUTPUT_DIR", "outputs")
    config = _get_config(
        base_model="ltx", n_gpus=16, bsz=2, gradient_step_per_epoch=1, dataset="vggsound", reward_fn=reward_fn, name=exp_name, prefix=prefix
    )

    config.pretrained.model = os.environ.get("LTX_MODEL_PATH", "checkpoints/ltx-2-19b-dev.safetensors")
    config.gemma_root = os.environ.get("GEMMA_MODEL_PATH", "checkpoints/gemma-3-12b-it")

    config.train.adv_mode = "gdpo"

    config.train.attn_sync_weight_max = 1

    config.train.learning_rate = 3e-5

    config.save_freq = 25
    config.eval_freq = 50

    config.use_lora = True
    config.target_modules = target_modules
    config.train.ema = False
    config.train.use_fsdp = True

    return config


def branch_aware_layer_surgery_avweight():
    reward_fn = {
        "videoalign_score": 1.0,
        "hpsv3_score_video": 1.5,
        "audiobox_aesthetics_score": 0.5,
        "clap_score": 1.0,
        "av_desync_reward": 1.0
    }

    exp_name = "branch_aware_layer_surgery_avweight"
    prefix = os.environ.get("OUTPUT_DIR", "outputs")
    config = _get_config(
        base_model="ltx", n_gpus=16, bsz=2, gradient_step_per_epoch=1, dataset="vggsound", reward_fn=reward_fn, name=exp_name, prefix=prefix
    )

    config.pretrained.model = "ltx-2-19b-dev.safetensors"
    config.gemma_root = "google/gemma-3-12b-it-qat-q4_0-unquantized"

    # adv routing
    config.train.adv_mode = "branch_aware"

    # gradients surgery
    config.train.ca_kv_scale_a2v = [
        {"blocks": [0],       "dir": "a2v", "scale": 0.0},
        {"blocks": ["1-10"],  "dir": "a2v", "scale": 0.1},
        {"blocks": ["40-47"], "dir": "a2v", "scale": 0.3},
    ]

    # Reweighting
    config.train.attn_sync_weight_max = 1.5
    config.train.attn_sync_warmup_steps = 400

    config.train.learning_rate = 3e-5

    config.save_freq = 25
    config.eval_freq = 50

    config.use_lora = True
    config.target_modules = target_modules
    config.train.ema = False
    config.train.use_fsdp = True

    return config