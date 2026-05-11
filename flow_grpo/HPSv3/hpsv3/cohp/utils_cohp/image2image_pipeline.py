from diffusers import FluxImg2ImgPipeline, KolorsImg2ImgPipeline, StableDiffusion3Img2ImgPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import torch
import os
class Image2ImagePipeline:
    def __init__(
        self, pipe_name, device='cuda'
    ):
        self.pipe_name = pipe_name
        if self.pipe_name == 'flux':
            self.pipeline = FluxImg2ImgPipeline.from_pretrained("pretrained_models/FLUX.1-dev",torch_dtype=torch.bfloat16).to(device)
            self.generation_path = 'generation/flux_dev',
        elif self.pipe_name == 'kolors':
            self.pipeline = KolorsImg2ImgPipeline.from_pretrained("/preflab/shuiyunhao/tasks/HPSv3/pretrained_models/kolors",torch_dtype=torch.bfloat16).to(device)
            self.generation_path = 'generation/kolors',

        elif self.pipe_name == 'sd3':
            self.pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium",torch_dtype=torch.bfloat16).to(device)
            self.generation_path = 'generation/sd3_medium',
        elif self.pipe_name == 'playground_v2_5':
            self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained("pretrained_models/playground-v2.5-1024px-aesthetic",torch_dtype=torch.bfloat16).to(device)
            self.generation_path = 'generation/playground_v_2_5',
        self.pipeline = self.pipeline.to(torch.bfloat16)
    def generate_image(
        self,
        prompt,
        image_path,
        strength,
        batch_size,
        save_prefix,
        output_dir
    ):
        image_load = load_image(image_path)
        if self.pipe_name == 'flux':
            images = self.pipeline(
            prompt = prompt,
            image=image_load, 
            num_images_per_prompt=batch_size, 
            strength = strength).images
        else:

            images = self.pipeline(
                prompt = prompt,
                negative_prompt = '',
                image=image_load, 
                num_images_per_prompt=batch_size, 
                strength = strength).images
        image_list = []
        for ind,img in enumerate(images):
            print(output_dir,self.generation_path,save_prefix)
            save_path = os.path.join(output_dir,self.generation_path[0],save_prefix+f'_{ind}.png')
            image_list.append(save_path)
            img.save(save_path)
        print(image_list)
        return image_list

# pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained("/preflab/shuiyunhao/tasks/HPSv3/pretrained_models/stable-diffusion-3-medium-diffusers",torch_dtype=torch.bfloat16).to('cuda:0')
# pipeline = pipeline.to(torch.bfloat16)
# image_load = load_image('/preflab/shuiyunhao/tasks/HPSv3/cohp_output/generation/flux_dev/0_origin_0.png')
# images = pipeline(
#             prompt = 'a girl',
#             negative_prompt = '',
#             image=image_load, 
#             num_images_per_prompt=1, 
#             strength = 0.8).images