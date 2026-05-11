import os
import json
import random
import gc
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from generator import Generator
from hpsv3.inference import HPSv3RewardInferencer
from hpsv3.cohp.utils_cohp.pipelines import *
from hpsv3.cohp.utils_cohp.image2image_pipeline import Image2ImagePipeline

try:
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
except:
    print("HPSv2 model not found, skipping HPSv2 related imports.")

try:
    import ImageReward as RM
except:
    print("ImageReward module not found, skipping ImageReward related imports.")


def initialize_hpsv2_model(device, checkpoint_path):
    model_dict = {}
    model, _, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        device=device,
        precision='amp',
        pretrained_image=False,
        output_dict=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()
    tokenizer = get_tokenizer('ViT-H-14')

    model_dict['model'] = model
    model_dict['preprocess_val'] = preprocess_val
    return model_dict, tokenizer


def score_hpsv2(model_dict, tokenizer, device, img_paths, prompts):
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']
    images = [preprocess_val(Image.open(p)).unsqueeze(0) for p in img_paths]
    images = torch.cat(images, dim=0).to(device)
    texts = tokenizer(prompts).to(device)
    
    with torch.no_grad():
        outputs = model(images, texts)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        hps_scores = torch.diagonal(logits_per_image).cpu()
    return hps_scores


def calculate_pickscore_probs(model, processor, prompt, images, device):
    image_inputs = processor(images=images, padding=True, return_tensors="pt").to(device)
    text_inputs = processor(text=prompt, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs /= torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs /= torch.norm(text_embs, dim=-1, keepdim=True)

        scores = text_embs @ image_embs.T
    return scores


def generate_images(
    reward_type, prompt, index, pipeline_params, pipelines_mapping, inferencer,
    output_dir='cohp_output', num_rounds=5, strength=0.8, device='cuda:1'
):
    os.makedirs(output_dir, exist_ok=True)
    result_json_dir = os.path.join(output_dir, 'result_json')
    os.makedirs(result_json_dir, exist_ok=True)

    info_dict = {
        'caption': prompt,
        'width': 1024,
        'height': 1024,
        'aspect_ratio': 1,
        'save_name': f"{index}_origin",
    }
    di_score_pipelines = {}
    intermediate_results_model_pref = {}
    intermediate_results_sample_pref = {}
    max_final_score = 0

    for pipeline_param in pipeline_params:
        generator = Generator(
            device=device,
            pipe_name=pipeline_param.pipeline_name,
            pipe_type=pipeline_param.pipeline_type,
            pipe_init_kwargs=pipeline_param.pipe_init_kwargs,
        )
        image_paths = generator.generate_imgs(
            info_dict=info_dict,
            generation_path=os.path.join(output_dir, pipeline_param.generation_path),
            batch_size=2,
            device=device,
            seed=random.randint(0, 75859066837),
            weight_dtype=pipeline_param.pipe_init_kwargs["torch_dtype"],
            generation_kwargs=pipeline_param.generation_kwargs

        )

        score_list = []
        for image_path in image_paths:
            if reward_type == 'hpsv2':
                score = score_hpsv2(model_dict, tokenizer, device, [image_path], [prompt]).item()
            elif reward_type == 'hpsv3':
                score = inferencer.reward([image_path], [prompt]).cpu().detach()[0][0].item()
            elif reward_type == 'imagereward':
                score = inferencer.score(prompt, [image_path])
            elif reward_type == 'pickscore':
                score = calculate_pickscore_probs(inferencer, processor_pickscore, prompt, [Image.open(image_path)], device)[0][0].item()
            else:
                raise ValueError(f"Unsupported reward type: {reward_type}")
            score_list.append(score)

        average_score = sum(score_list) / len(score_list)
        pipeline_name = pipelines_mapping[pipeline_param]
        di_score_pipelines[pipeline_name] = average_score

        intermediate_results_model_pref[pipeline_name] = {
            'image_paths': image_paths,
            'scores': score_list,
            'max_image_path': image_paths[score_list.index(max(score_list))],
            'max_score': max(score_list),
        }
        generator.pipelines.to("cpu")
        del generator
        torch.cuda.empty_cache()
        gc.collect()

    # Select the best pipeline based on scores
    best_pipeline = max(di_score_pipelines, key=di_score_pipelines.get)
    best_pipeline_results = intermediate_results_model_pref[best_pipeline]
    chosen_image_path = best_pipeline_results['max_image_path']

    # Refinement with Image2ImagePipeline
    i2ipipeline = Image2ImagePipeline(best_pipeline)
    for round_num in range(num_rounds):
        if round_num in [3, 4]:
            strength = 0.5
        images = i2ipipeline.generate_image(
            prompt=prompt,
            image_path=chosen_image_path,
            strength=strength,
            batch_size=4,
            save_prefix=f'{index}_{best_pipeline}_image2image_round{round_num + 1}',
            output_dir=output_dir,
        )

        score_list = []
        for image_path in images:
            if reward_type == 'hpsv2':
                score = score_hpsv2(model_dict, tokenizer, device, [image_path], [prompt]).item()
            elif reward_type == 'hpsv3':
                score = inferencer.reward([image_path], [prompt]).cpu().detach()[0][0].item()
            elif reward_type == 'imagereward':
                score = inferencer.score(prompt, [image_path])
            elif reward_type == 'pickscore':
                score = calculate_pickscore_probs(inferencer, processor_pickscore, prompt, [Image.open(image_path)], device)[0][0].item()
            else:
                raise ValueError(f"Unsupported reward type: {reward_type}")
            score_list.append(score)

        # Update intermediate results
        intermediate_results_sample_pref[round_num + 1] = {
            'image_paths': images,
            'scores': score_list,
            'max_image_path': images[score_list.index(max(score_list))],
            'max_score': max(score_list),
        }

        # Determine best image during refinement
        if max(score_list) > max_final_score:
            max_final_score = max(score_list)
            chosen_image_path = images[score_list.index(max(score_list))]

    # Save final results
    results = {
        'prompt': prompt,
        'best_model': best_pipeline,
        'final_image_path': chosen_image_path,
        'model_preference_info': intermediate_results_model_pref,
        'sample_preference_intermediate_results': intermediate_results_sample_pref,
    }
    with open(os.path.join(result_json_dir, f'{index}.json'), 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Generation Script")
    parser.add_argument('--prompt', type=str, required=True, help='The prompt for image generation')
    parser.add_argument('--index', type=str, required=True, help='Index for saving results')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on')
    parser.add_argument('--reward_model', type=str, default='hpsv3', help='Reward model to use (hpsv2, hpsv3, pickscore, or imagereward)')
    args = parser.parse_args()

    # Initialize models and pipelines
    output_dir = f"cohp_output_{args.reward_model}"
    if args.reward_model == 'hpsv2':
        model_dict, tokenizer = initialize_hpsv2_model(args.device, 'pretrained_models/HPS_v2.1_compressed.pt')
        inferencer = model_dict
    elif args.reward_model == 'hpsv3':
        inferencer = HPSv3RewardInferencer(device=args.device)
    elif args.reward_model == 'imagereward':
        inferencer = RM.load("ImageReward-v1.0").to(args.device)
    elif args.reward_model == 'pickscore':
        processor_pickscore = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        inferencer = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(args.device)
    else:
        raise ValueError("Unsupported reward model.")

    # Define pipelines
    pipeline_params = [kolors_pipe, sd3_medium_pipe, playground_v2_5_pipe, flux_dev_pipe]
    pipelines_mapping = {
        flux_dev_pipe: 'flux',
        kolors_pipe: 'kolors',
        sd3_medium_pipe: 'sd3',
        playground_v2_5_pipe: 'playground_v2_5',
    }

    # Generate images
    results = generate_images(
        reward_type=args.reward_model,
        prompt=args.prompt,
        index=args.index,
        pipeline_params=pipeline_params,
        pipelines_mapping=pipelines_mapping,
        inferencer=inferencer,
        output_dir=output_dir,
        num_rounds=4,
    )