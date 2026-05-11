import torch
class PipelineParam:
    pipeline_name: str
    pipeline_type: str
    generation_path: str
    pipe_init_kwargs: dict
    generation_kwargs: dict
    base_resolution: int
    force_aspect_ratio: int
    
    def __init__(self, pipeline_name: str, generation_path: str, pipeline_type = 't2i',
                 pipe_init_kwargs: dict = None, generation_kwargs: dict = None,
                 base_resolution: int = 1024, force_aspect_ratio: int = None):
        self.pipeline_name = pipeline_name
        self.pipeline_type = pipeline_type
        self.generation_path = generation_path
        self.pipe_init_kwargs = pipe_init_kwargs if pipe_init_kwargs is not None else {}
        self.generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
        self.base_resolution = base_resolution
        self.force_aspect_ratio = force_aspect_ratio

flux_dev_pipe = PipelineParam(
        pipeline_name='pretrained_models/FLUX.1-dev',
        generation_path=f'generation/flux_dev',
        pipe_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        base_resolution=1024,
        generation_kwargs={
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "max_sequence_length": 512,
        }
    )

flux_schnell_pipe = PipelineParam(
        pipeline_name='/mnt2/share/huggingface_models/FLUX.1-schnell',
        generation_path=f'generation/flux_schnell',
        pipe_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        base_resolution=1024,
        generation_kwargs={
            "guidance_scale": 3.5,
            "num_inference_steps": 4,
        }
    )


sd3_medium_pipe = PipelineParam(
        pipeline_name='pretrained_models/stable-diffusion-3-medium-diffusers',
        generation_path=f'generation/sd3_medium',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=1024,
        generation_kwargs={
            "guidance_scale": 7.0,
            "num_inference_steps": 28,
        }
    )

sd_xl_pipe = PipelineParam(
        pipeline_name='pretrained_models/stable-diffusion-xl-base-1.0',
        generation_path=f'generation/sd_xl',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=1024,
        generation_kwargs={
            "guidance_scale": 5,
            "num_inference_steps": 50,
        }
    )

sd_1_5_pipe = PipelineParam(
        pipeline_name='pretrained_models/stable-diffusion-v1-5',
        generation_path=f'generation/sd_1_5',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=512,
        generation_kwargs={
        }
    )

vq_diffusion_pipe = PipelineParam(
        pipeline_name='pretrained_models/vq-diffusion-ithq',
        generation_path=f'generation/vq_diffusion',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=256,
        generation_kwargs={}
    )

sd_2_pipe = PipelineParam(
        pipeline_name='pretrained_models/stable-diffusion-2',
        generation_path=f'generation/sd_2',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=512,
        force_aspect_ratio=1,
    )

sd_1_1_pipe = PipelineParam(
        pipeline_name='pretrained_models/stable-diffusion-v1-1',
        generation_path=f'generation/sd_1_1',
        pipe_init_kwargs={"torch_dtype": torch.float16,},
        base_resolution=512,
        force_aspect_ratio=1,
    )

sd_1_4_pipe = PipelineParam(
        pipeline_name='pretrained_models/stable-diffusion-v1-4',
        generation_path=f'generation/sd_1_4',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=512,
        force_aspect_ratio=1,
    )

sd_2_1_pipe = PipelineParam(
        pipeline_name='pretrained_models/stable-diffusion-2-1-base',
        generation_path=f'generation/sd_2_1',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=512,
        force_aspect_ratio=1,
    )

openjourney_pipe = PipelineParam(
        pipeline_name='pretrained_models/openjourney',
        generation_path=f'generation/openjourney',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=512,
        force_aspect_ratio=1,
    )

playground_v2_5_pipe = PipelineParam(
        pipeline_name='pretrained_models/playground-v2.5-1024px-aesthetic',
        generation_path=f'generation/playground_v_2_5',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=1024,
    )

versatile_pipe = PipelineParam(
        pipeline_name='pretrained_models/versatile-diffusion',
        generation_path=f'generation/versatile',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=512,
        force_aspect_ratio=1,
    )

glide_pipe = PipelineParam(
        pipeline_name='pretrained_models/glide-base',
        generation_path=f'generation/glide',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=512,
        force_aspect_ratio=1,
)

sd_3_5_medium_pipe = PipelineParam(
        pipeline_name='stabilityai/stable-diffusion-3.5-medium',
        generation_path=f'generation/sd_3_5_medium',
        pipe_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        base_resolution=1024,
        generation_kwargs={
            "num_inference_steps": 40,
            "guidance_scale": 4.5,
        }
    )

sd_3_5_large_pipe = PipelineParam(
        pipeline_name='stabilityai/stable-diffusion-3.5-large',
        generation_path=f'generation/sd_3_5_large',
        pipe_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        base_resolution=1024,
        generation_kwargs={
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
        }
    )

kolors_pipe = PipelineParam(
        pipeline_name='pretrained_models/Kolors-diffusers',
        generation_path=f'generation/kolors',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
            'variant': 'fp16',
        },
        base_resolution=1024,
        generation_kwargs={
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
        }
    )

cogview4_pipe = PipelineParam(
        pipeline_name='pretrained_models/CogView4-6B',
        generation_path=f'generation/cogview4',
        pipe_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        base_resolution=1024,
        generation_kwargs={
            "num_inference_steps": 50,
            "guidance_scale": 3.5,
        }
    )

pixart_sigma_pipe = PipelineParam(
        pipeline_name='pretrained_models/PixArt-Sigma-XL-2-1024-MS',
        generation_path=f'generation/pixart_sigma',
        pipeline_type='t2i',
        pipe_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        base_resolution=1024,
)

hunyuanvideo_pipe = PipelineParam(
        pipeline_name='pretrained_models/hunyuanvideo_diffusers',
        generation_path=f'generation/hunyuanvideo',
        pipe_init_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        base_resolution=1024,
        pipeline_type='t2v',
        generation_kwargs={
            "num_inference_steps": 30,
            "num_frames": 1,
        }
)

hunyuandit_pipe = PipelineParam(
        pipeline_name='pretrained_models/HunyuanDiT-v1.2-Diffusers',
        generation_path=f'generation/hunyuandit',
        pipe_init_kwargs={
            "torch_dtype": torch.float16,
        },
        base_resolution=1024,
        pipeline_type='t2i',
        generation_kwargs={
        }
)

# API models
# Fal.ai
flux_pro_v1_1_ultr_pipe = PipelineParam(
        pipeline_name='fal-ai/flux-pro/v1.1-ultra',
        generation_path=f'generation/flux_pro_v1_1_ultra',
        base_resolution=1024,
        generation_kwargs={
            "enable_safety_checker": False,
            "num_images": 1,
            # "aspect_ratio": "1:1",
            "output_format": "jpeg",
            "safety_tolerance": 5,
        }
    )

recraftv3_pipe = PipelineParam(
        pipeline_name='fal-ai/recraft-v3',
        generation_path=f'generation/recraftv3',
        base_resolution=1024,
        generation_kwargs={
            "enable_safety_checker": False,
            "num_images": 1,
            # "aspect_ratio": "1:1",
            "output_format": "jpeg",
            "safety_tolerance": 5,
        }
    )

