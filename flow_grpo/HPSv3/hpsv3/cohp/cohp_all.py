import os
from generator import Generator
import json
import os
import torch
import gc
from utils_cohp.pipelines import *
from utils_cohp.image2image_pipeline import Image2ImagePipeline
import argparse
from ..inference import HPSv3RewardInferencer
import random
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import ImageReward as RM
from PIL import Image
from transformers import AutoProcessor, AutoModel

def initialize_model(device, cp):
    model_dict = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )

    checkpoint = torch.load(cp, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer('ViT-H-14')

    model_dict['model'] = model
    model_dict['preprocess_val'] = preprocess_val
    return model_dict, tokenizer

def score_hpsv2_batch(model_dict, tokenizer, device, img_paths: list, prompts: list) -> list:
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']
    # 批量处理图片
    images = [preprocess_val(Image.open(p)).unsqueeze(0) for p in img_paths]
    images = torch.cat(images, dim=0).to(device=device)
    texts = tokenizer(prompts).to(device=device)
    with torch.no_grad():
        outputs = model(images, texts)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        hps_scores = torch.diagonal(logits_per_image).cpu()
    return hps_scores
def pickscorecalc_probs(model,processor_pickscore,prompt, images, device):
    
    # preprocess
    image_inputs = processor_pickscore(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor_pickscore(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = text_embs @ image_embs.T
        
    return scores

def generate_images(reward_type, prompt, index, pipeline_params, di_pipeline, inferencer, out_dir='cohp_output', num_rounds=5, strength=0.8, device='cuda:1'):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'result_json'), exist_ok=True)
    batch_size = 2  # 设置batch大小

    results = []  # 用于保存每个 prompt 的最终结果

    
    info_dict = {
        'caption': prompt,
        'width': 1024,
        'height': 1024,
        'aspect_ratio': 1,
        'save_name': f"{index}_origin",
    }
    di_score_pipelines = {}  # 用于存储 pipeline 的平均分数

    # 中间结果记录结构：用于保存每一轮图像路径和分数
    intermediate_results_sample_preference = []
    intermediate_results_model_preference = []

    # 遍历 pipeline 参数
    for pipeline_param in pipeline_params:
        
        name = di_pipeline[pipeline_param]
        generator = Generator(
            device = device,
            pipe_name=pipeline_param.pipeline_name,
            pipe_type=pipeline_param.pipeline_type,
            pipe_init_kwargs=pipeline_param.pipe_init_kwargs,
        )
        image_paths = generator.generate_imgs(
            info_dict = info_dict,
            generation_path = os.path.join(out_dir, pipeline_param.generation_path),
            batch_size=batch_size,
            device = device,
            seed=random.randint(0, 75859066837),
            weight_dtype=pipeline_param.pipe_init_kwargs["torch_dtype"],
            generation_kwargs=pipeline_param.generation_kwargs,
        )

        # 对生成的图像进行评分
        score_list = []
        for image_path in image_paths:
            if reward_type == 'hpsv2':
                score = score_hpsv2_batch(model_dict, tokenizer, device, [image_path], [prompt])
                score = score.item()
            elif reward_type == 'hpsv3':
                score = inferencer.reward([image_path], [prompt]).cpu().detach()
                score = score[0][0].item()
            elif reward_type == 'imagereward':
                score = inferencer.score(prompt, [image_path])
            elif reward_type == 'pickscore':
                score = pickscorecalc_probs(inferencer, processor_pickscore, prompt, [Image.open(image_path)],device)[0][0].item()
                print(f"PickScore for {image_path}: {score}")
            else:
                raise ValueError("Unsupported reward type. Choose 'hpsv2', 'hpsv3', or 'imagereward'.")
            score_list.append(score)

        average = sum(score_list) / len(score_list)
        di_score_pipelines[name] = average
        # 保存中间步骤的图像路径和分数
        intermediate_results_model_preference.append({
            'pipeline': name,
            'image_paths': image_paths,  # 所有生成的图片路径
            'scores': score_list,  # 每轮的得分列表
            'max_image_path': image_paths[score_list.index(max(score_list))],  # 当前轮得分最高的图片路径
            'max_score': max(score_list)  # 当前轮得分最高的分数
        })
        
        # 清理生成器资源
        generator.pipelines.to("cpu")
        del generator
        torch.cuda.empty_cache()
        gc.collect()

    # 选择得分最高的 pipeline 和对应的图片
    max_key = max(di_score_pipelines, key=di_score_pipelines.get)
    max_index = score_list.index(max(score_list))
    image_path_chosen = image_paths[max_index]  # 首轮选择的最佳图片

    # 多轮优化循环

    for round_num in range(num_rounds):
        if round_num == 3 or round_num == 4:
            strength = 0.5
        i2ipipeline = Image2ImagePipeline(max_key)
        images = i2ipipeline.generate_image(
            prompt=prompt,
            image_path=image_path_chosen,
            strength=strength,
            batch_size=4,
            save_prefix=f'{index}_{max_key}_image2image_round{round_num + 1}',
            output_dir=out_dir
        )

        score_list = []
        for image_path in images:
            if reward_type == 'hpsv2':
                score = score_hpsv2_batch(model_dict, tokenizer, device, [image_path], [prompt])
                score = score.item()
            elif reward_type == 'hpsv3':
                score = inferencer.reward([image_path], [prompt]).cpu().detach()
                score = score[0][0].item()
            elif reward_type == 'imagereward':
                score = inferencer.score(prompt, [image_path])
            elif reward_type == 'pickscore':
                score = pickscorecalc_probs(inferencer, processor_pickscore, prompt, [Image.open(image_path)],device)[0][0].item()
                print(f"PickScore for {image_path}: {score}")
            else:
                raise ValueError("Unsupported reward type. Choose 'hpsv2', 'hpsv3', or 'imagereward'.")
            score_list.append(score)

        intermediate_results_sample_preference.append({
            'round': round_num + 1,
            'image_paths': images,  # 所有生成的图片路径
            'scores': score_list,  # 每轮的得分列表
            'max_image_path': images[score_list.index(max(score_list))],  # 当前轮得分最高的图片路径
            'max_score': max(score_list)  # 当前轮得分最高的分数
        })

        # 更新图片选择
        max_index = score_list.index(max(score_list))
        image_path_chosen = images[max_index]
    


    # 最终结果保存
    results.append({
        'prompt': prompt,
        'model_preference_image_chosen': image_path_chosen,
        "model_preference_info": intermediate_results_model_preference,  # 包含所有中间结果
        'best_image_path': image_path_chosen,
        'best_model': max_key,
        'score': max(score_list),
        'sample_preference_intermediate_results': intermediate_results_sample_preference,  # 包含所有中间结果
    })
    with open(os.path.join(out_dir, 'result_json',f'{index}.json'),'w',encoding='utf-8') as f:
        json.dump(results,f,ensure_ascii=False, indent=4)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Generation Script")
    parser.add_argument('--prompt', type=str, required=True, help='The prompt for image generation')
    parser.add_argument('--index', type=str, required=True, help='Index for saving results')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on')
    parser.add_argument('--reward_model', type=str, default='hpsv3', help='Reward model to use (hpsv2 or hpsv3 or pickscore or imagereward)')
    
    args = parser.parse_args()
    output_dir = f"cohp_output_{args.reward_model}"

    os.makedirs(output_dir,exist_ok=True)
    if args.reward_model == 'hpsv2':
        
        inferencer = initialize_model(args.device, 'pretrained_models/HPS_v2.1_compressed.pt')
        model_dict, tokenizer = inferencer
    elif args.reward_model == 'hpsv3':
        dtype = torch.bfloat16
        inferencer = HPSv3RewardInferencer(device=args.device, dtype=dtype)
    elif args.reward_model == 'imagereward':
        inferencer = RM.load("ImageReward-v1.0").to(args.device)
    elif args.reward_model == 'pickscore':
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        processor_pickscore = AutoProcessor.from_pretrained(processor_name_or_path)
        inferencer = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(args.device)
    else:
        raise ValueError("Unsupported reward model. Choose 'hpsv2', 'hpsv3', or 'imagereward'.")
    pipeline_params = [
                flux_dev_pipe,
                kolors_pipe,
                sd3_medium_pipe,
                playground_v2_5_pipe
            ]

    di_score_pipelines={}
    di_pipeline = {
        flux_dev_pipe:'flux',
        kolors_pipe:'kolors',
        sd3_medium_pipe:'sd3',
        playground_v2_5_pipe:'playground_v2_5'

    }

    results = generate_images(
            args.reward_model,
            args.prompt, 
            args.index, 
            pipeline_params, 
            di_pipeline, 
            inferencer, 
            out_dir=output_dir, 
            num_rounds=4, 
            strength=0.8,
            device=args.device)
