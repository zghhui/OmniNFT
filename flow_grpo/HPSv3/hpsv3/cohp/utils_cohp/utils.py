import torch
try:
    import fal_client
except:
    fal_client = None
try:
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
except:
    AutoPipelineForText2Image = None
    DiffusionPipeline = None 
    
import json
import diffusers
from functools import partial
import os
# export FAL_KEY="YOUR_API_KEY"
os.environ['FAL_KEY'] = 'YOUR_API_KEY'

def init_pipelines(pipe_name, pipe_init_kwargs, device=None):

    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(pipe_name, **pipe_init_kwargs).to(device)
    except Exception as e:
        # try:
            config = json.load(open(os.path.join(pipe_name, 'model_index.json')))
            class_name_str = config['_class_name']
            pipeline_class = getattr(diffusers, class_name_str)
            pipeline = pipeline_class.from_pretrained(pipe_name, **pipe_init_kwargs).to(device)

    return pipeline


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