import torch
import os
import inspect
from PIL import Image
from tqdm import tqdm
from utils_cohp.utils import init_pipelines

Image.MAX_IMAGE_PIXELS = None


class Generator:
    def __init__(
        self, pipe_name, pipe_type, pipe_init_kwargs, device=None
    ):
        self.pipe_names = pipe_name
        self.pipe_type = pipe_type
        self.pipe_init_kwargs = pipe_init_kwargs
        self.pipelines = init_pipelines(
            pipe_name, pipe_init_kwargs, device
        )

    def generate_imgs(
        self,
        batch_size,
        generation_path,
        info_dict,
        device,
        weight_dtype,
        seed,
        generation_kwargs,

    ):

        torch.cuda.set_device(device)
        device = torch.device(device)
        generator = torch.Generator().manual_seed(seed)
        
        pipeline_signature = inspect.signature(self.pipelines)
        pipeline_params = pipeline_signature.parameters.keys()
        
        if 'height' not in pipeline_params:
            generation_kwargs.pop('height', None)
            print(f"Warning: Pipeline does not support 'height' parameter, removing from kwargs")
        if 'width' not in pipeline_params:
            generation_kwargs.pop('width', None)
            print(f"Warning: Pipeline does not support 'width' parameter, removing from kwargs")
        

        outputs = self.pipelines(
            prompt=info_dict['caption'], generator=generator,num_images_per_prompt = batch_size, **generation_kwargs
        )
        if self.pipe_type == "t2i":
            images = outputs.images
        elif self.pipe_type == "t2v":
            images = outputs.frames[0]
        image_paths = []
        for idx, image in enumerate(images):            
            img_path = os.path.join(
                generation_path, info_dict["save_name"] + f"_{idx}.png"
            ) 
            os.makedirs(generation_path,exist_ok=True)         
            image.save(img_path)
            image_paths.append(img_path)
        return image_paths