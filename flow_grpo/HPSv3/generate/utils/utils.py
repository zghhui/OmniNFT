import torch
try:
    import fal_client
except:
    fal_client = None

from diffusers import AutoPipelineForText2Image, HunyuanVideoPipeline, DiffusionPipeline
import json
import diffusers
from functools import partial
import os
# export FAL_KEY="YOUR_API_KEY"
os.environ['FAL_KEY'] = 'YOUR_API_KEY'

def init_multiple_pipelines(pipe_name, pipe_init_kwargs, num_devices, device_id=None):
    pipelines_dict = []

    if device_id is not None:
        assert num_devices == 1

    for i in range(num_devices):
        actual_device_id = device_id if device_id is not None else i
        try:
            pipeline = AutoPipelineForText2Image.from_pretrained(pipe_name, **pipe_init_kwargs).to(f'cuda:{actual_device_id}')
        except Exception as e:
            # try:
                config = json.load(open(os.path.join(pipe_name, 'model_index.json')))
                class_name_str = config['_class_name']
                pipeline_class = getattr(diffusers, class_name_str)
                pipeline = pipeline_class.from_pretrained(pipe_name, **pipe_init_kwargs).to(f'cuda:{actual_device_id}')
        # except Exception as ew:
        #     print(e)
        #     pipeline = DiffusionPipeline.from_pretrained(pipe_name, **pipe_init_kwargs).to(f'cuda:{actual_device_id}')
        pipelines_dict.append(pipeline)
    return pipelines_dict


def init_pipeline_from_names(pipe_names, weight_dtype):
    pipelines_dict = {}
    for name in pipe_names:
        pipeline = AutoPipelineForText2Image.from_pretrained(name, torch_dtype=weight_dtype)
        pipelines_dict[name] = pipeline
    return pipelines_dict


def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])

def gen_with_api(pipe_names, generation_kwargs):
    result = fal_client.subscribe(
        pipe_names,
        arguments=generation_kwargs,
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    return result