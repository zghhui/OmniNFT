import torch
import os
import inspect
from PIL import Image
from tqdm import tqdm
from utils.utils import init_multiple_pipelines
from concurrent.futures import ThreadPoolExecutor, as_completed

Image.MAX_IMAGE_PIXELS = None


class Generator:
    def __init__(
        self, pipe_name, pipe_type, pipe_init_kwargs, num_devices, device_id=None
    ):
        self.pipe_names = pipe_name
        self.pipe_type = pipe_type
        self.pipe_init_kwargs = pipe_init_kwargs
        self.pipelines = init_multiple_pipelines(
            pipe_name, pipe_init_kwargs, num_devices, device_id
        )

    def generate_imgs(
        self,
        num_device,
        batch_size,
        generation_path,
        info_dict,
        pipeline,
        device_id,
        weight_dtype,
        seed,
        base_resolution,
        force_aspect_ratio,
        generation_kwargs,

    ):

        torch.cuda.set_device(f"cuda:{device_id%num_device}")
        device = torch.device(f"cuda:{device_id%num_device}")

        num_prompts_per_device = len(info_dict) // num_device
        start_idx = device_id * num_prompts_per_device
        end_idx = (
            start_idx + num_prompts_per_device
            if device_id != (num_device - 1)
            else len(info_dict)
        )

        device_info_dict = info_dict[start_idx:end_idx]

        print(f"Device {device} generating for prompts {start_idx} to {end_idx-1}")

        print("## Prepare generation dataset")

        total_batches = len(device_info_dict) // batch_size + (
            1 if len(device_info_dict) % batch_size != 0 else 0
        )
        for batch_idx in tqdm(
            range(total_batches), desc="Pipeline: " + self.pipe_names
        ):
            batch_info_dict = device_info_dict[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            save_paths = []
            for info_dict in batch_info_dict:
                if info_dict["image_file"] is not None:
                    save_paths.append(
                        os.path.join(generation_path, info_dict["image_file"][:-4] + ".png")
                    )
                else:
                    save_paths.append(
                        os.path.join(generation_path, info_dict["save_name"] + ".png")
                    )

            exists_idx = []
            for i, save_path in enumerate(save_paths):
                if os.path.exists(save_path):
                    exists_idx.append(i)

            batch_info_dict = [
                batch_info_dict[i]
                for i in range(len(batch_info_dict))
                if i not in exists_idx
            ]
            if len(batch_info_dict) == 0:
                continue

            batch_prompts = [info_dict["caption"] for info_dict in batch_info_dict]
            batch_image_file = [
                info_dict["image_file"] for info_dict in batch_info_dict
            ]
            if batch_image_file[0] is not None:
                try:
                    batch_image_sizes = [
                        Image.open(image_file).size for image_file in batch_image_file
                    ]
                except:
                    batch_image_sizes = None
            else:
                batch_image_sizes = [
                    (batch_info_dict[i]["width"], batch_info_dict[i]["height"])
                    for i in range(len(batch_info_dict))
                ]

            if batch_image_sizes is None:
                aspect_ratios = [
                    info_dict["aspect_ratio"] for info_dict in batch_info_dict
                ]
            else:
                aspect_ratios = [size[0] / size[1] for size in batch_image_sizes]

            if force_aspect_ratio:
                height = int(base_resolution / force_aspect_ratio // 64 * 64)
                width = int(base_resolution * force_aspect_ratio // 64 * 64)
            else:
                # 根据aspect_ratios调整base_resolution, 得到height和width， 保证调整后的乘积大概等于base_resolution**2
                height = int(base_resolution / aspect_ratios[0] ** (0.5) // 64 * 64)
                width = int(base_resolution * aspect_ratios[0] ** (0.5) // 64 * 64)
            generation_kwargs.update({"height": height, "width": width})

            generator = torch.Generator().manual_seed(seed + batch_idx)
            
            pipeline_signature = inspect.signature(pipeline)
            pipeline_params = pipeline_signature.parameters.keys()
            
            if 'height' not in pipeline_params:
                generation_kwargs.pop('height', None)
                print(f"Warning: Pipeline does not support 'height' parameter, removing from kwargs")
            if 'width' not in pipeline_params:
                generation_kwargs.pop('width', None)
                print(f"Warning: Pipeline does not support 'width' parameter, removing from kwargs")
            
            try:
                outputs = pipeline(
                    prompt=batch_prompts, generator=generator, **generation_kwargs
                )
            except Exception as e:
                print(e)
                continue
            if self.pipe_type == "t2i":
                images = outputs.images
            elif self.pipe_type == "t2v":
                images = outputs.frames[0]

            for img_idx, (img, prompt, image_file, info_dict) in enumerate(
                zip(images, batch_prompts, batch_image_file, batch_info_dict)
            ):
                if image_file is None:
                    img_path = os.path.join(
                        generation_path, info_dict["save_name"] + ".png"
                    )
                else:
                    img_path = generation_path + image_file[:-4] + ".png"

                if not os.path.exists(os.path.dirname(img_path)):
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                img.save(img_path)
                if image_file is None:
                    text_path = os.path.join(
                        generation_path, info_dict["save_name"] + ".txt"
                    )
                else:
                    text_path = generation_path + image_file[:-4] + ".txt"
                try:
                    with open(text_path, "w") as f:
                        f.write(prompt)
                        f.write("\n")
                        f.write(
                            image_file
                            if image_file is not None
                            else info_dict["save_name"]
                        )
                except:
                    pass
        return True

    def generate(
        self,
        info_dict,
        generation_path,
        num_processes,
        batch_size,
        weight_dtype,
        seed,
        generation_kwargs,
        base_resolution,
        force_aspect_ratio,
    ):

        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(
                    self.generate_imgs,
                    num_processes,
                    batch_size,
                    generation_path,
                    info_dict,
                    self.pipelines[device_id],
                    device_id,
                    weight_dtype,
                    seed,
                    base_resolution,
                    force_aspect_ratio,
                    generation_kwargs,
                )
                for device_id in range(num_processes)
            ]

            for future in as_completed(futures):
                print(f"Task completed: {future.result()}")
