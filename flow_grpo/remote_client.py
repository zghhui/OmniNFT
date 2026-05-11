import torch
import base64
import io
import requests
from PIL import Image
from torchvision import transforms
from typing import Union, List
import re
import difflib
from itertools import groupby
import re
from collections import Counter
import time



def smart_tokenize(text: str) -> list:
    """
    智能分词：
    1. 英文/数字：保持连续，如 "apple" -> "apple", "user_123" -> "user_123"
    2. 中文：按字切分，如 "你好" -> "你", "好"
    3. 标点：单独切分
    """
    # 正则解释：
    # [\u4e00-\u9fa5]       : 匹配单个汉字
    # [a-zA-Z0-9_]+         : 匹配连续的英文、数字、下划线
    # [^\s]                 : 匹配其他非空白字符（标点等），作为兜底
    
    # 查找所有匹配项
    # 注意：这个正则优先匹配汉字，然后是英文单词，最后是其他符号
    pattern = r'[\u4e00-\u9fa5]|[a-zA-Z0-9_]+|[^\s\w]'
    
    return re.findall(pattern, text)
def _pil_to_base64(pil_img):
    """辅助函数：PIL Image 转 Base64 字符串"""
    buffer = io.BytesIO()
    # 强制转 RGB 防止 RGBA 等格式报错
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def pre_process(prompts: Union[str, List[str]], images: Union[torch.Tensor, List[str], List[Image.Image]]):
    # --- 1. 统一 Prompts 格式为 List[str] ---
    if isinstance(prompts, str):
        # 如果后面 images 是 Tensor，这里先不扩展，等拿到 bsz 再扩展
        prompts_list = [prompts]
    else:
        prompts_list = prompts

    # --- 2. 统一 Images 格式为 List[base64_str] ---
    encoded_images = []
    
    # 情况 A: 输入是 torch.Tensor [bsz, c, h, w]
    if isinstance(images, torch.Tensor):
        img_tensor = images.detach().cpu()
        if img_tensor.ndim == 3: # 单张图转为 batch 格式处理
            img_tensor = img_tensor.unsqueeze(0)
        
        bsz = img_tensor.shape[0]
        to_pil = transforms.ToPILImage()
        for i in range(bsz):
            pil_img = to_pil(img_tensor[i])
            encoded_images.append(_pil_to_base64(pil_img))
            
        # 自动补全 prompts 长度
        if len(prompts_list) == 1 and bsz > 1:
            prompts_list = prompts_list * bsz

    # 情况 B: 输入是 List
    elif isinstance(images, list):
        for item in images:
            if isinstance(item, str):  # 路径
                with open(item, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    encoded_images.append(b64)
            elif isinstance(item, Image.Image): # PIL 对象
                encoded_images.append(_pil_to_base64(item))
            else:
                raise ValueError(f"Unsupported list item type: {type(item)}")
    else:
        raise ValueError(f"Unsupported images type: {type(images)}")

    # --- 3. 最终校验与发送 ---
    if len(prompts_list) != len(encoded_images):
        return {"error": f"Size mismatch: {len(prompts_list)} prompts vs {len(encoded_images)} images"}

    return prompts_list, encoded_images
    


def get_qwen_vl_reward_score(prompts: Union[str, List[str]], 
                             images: Union[torch.Tensor, List[str], List[Image.Image]], 
                             ip='6.178.129.113', port=80):
    prompts_list, encoded_images = pre_process(prompts, images)
    
    url = f"http://{ip}:{port}/predict"
    payload = {
        "prompts": prompts_list,
        "images_base64": encoded_images
    }

    try:
        # QwenVL 推理耗时较长（包含 Thought 过程），timeout 建议设为 120s
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        # 返回与 HPSv3 一致的 torch.tensor 格式
        return torch.tensor(response.json()['scores'])
    
    except Exception as e:
        if 'response' in locals() and response.status_code == 422:
            return {"error": "422 Validation Error", "detail": response.json()}
        return {"error": str(e)}




    
def get_ocr_vl_reward_score(
    prompts: Union[str, List[str]],
    images: Union[torch.Tensor, List[str], List[Image.Image]],
    target_texts: Union[str, List[str]],    
    ip: str = "6.178.129.78",
    port: int = 6024,
) -> Union[torch.Tensor, dict]:
    prompts_list, encoded_images = pre_process(prompts, images)
    url = f"http://{ip}:{port}/predict"
    payload = {
        "prompts": prompts_list,
        "images_base64": encoded_images,
        "target_texts": target_texts if isinstance(target_texts, list) else [target_texts] * len(prompts_list)
    }

    response = None
    try:
        response = requests.post(url, json=payload, timeout=1000)
        response.raise_for_status()
        return torch.tensor(response.json()["scores"])
    except Exception as e:
        print(f"error: {e}")
        if response is not None and response.status_code == 422:
            return {"error": "422 Validation Error", "detail": response.json()}
        return {"error": str(e)}


def get_hpsv3_reward_score(
    prompts: Union[str, List[str]], 
    images: Union[torch.Tensor, List[str], List[Image.Image]], 
    ip='6.178.129.113', port=80, token="<TOKEN>"
):
    prompts_list, encoded_images = pre_process(prompts, images)
    url = f"http://{ip}:{port}/predict"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
        }
    payload = {
        "prompts": prompts_list,
        "images_base64": encoded_images
    }

    # try:
    #     response = requests.post(url, json=payload, timeout=60)
    #     response.raise_for_status()
    #     return torch.tensor(response.json()['scores'])
    # except Exception as e:
    #     if response is not None and response.status_code == 422:
    #         return {"error": "422 Validation Error", "detail": response.json()}
    #     return {"error": str(e)}

    response = None  # 1. 显式初始化，避免 UnboundLocalError
    max_retries = 8
    
    for attempt in range(max_retries):
        try:
            # 增加 verify=False 如果是内网自签名证书
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return torch.tensor(response.json()['scores'])
            
        except Exception as err:
            print(f"get_hpsv3_reward_score error: {err} at attempt: {attempt}")
            time.sleep(1) # 普通错误等待后重试
            





import math
from typing import List, Union, Any
# ==========================================
# 0. 底层辅助函数 (核心算法)
# ==========================================
def _levenshtein_distance_generic(seq1: Union[str, List[Any]], seq2: Union[str, List[Any]]) -> int:
    """计算编辑距离的通用核心算法"""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    previous_row = range(size_y)
    for i, x in enumerate(seq1):
        current_row = [i + 1]
        for j, y in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if x == y else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def _compute_single_char_score(pred: str, target: str) -> float:
    """计算单条数据的字符分数"""
    if not pred and not target: return 1.0
    dist = _levenshtein_distance_generic(pred, target)
    max_len = max(len(pred), len(target))
    return 1.0 - (dist / max_len) if max_len > 0 else 0.0

def _compute_single_word_score(pred: str, target: str) -> float:
    """
    计算单词分数的改进版：支持中文按字切分，英文按词切分。
    """
    # 使用智能分词，而不是 split()
    pred_tokens = smart_tokenize(pred)
    target_tokens = smart_tokenize(target)
    
    if not pred_tokens and not target_tokens: return 1.0
    
    # 计算 token 列表的编辑距离
    dist = _levenshtein_distance_generic(pred_tokens, target_tokens)
    max_len = max(len(pred_tokens), len(target_tokens))
    
    return 1.0 - (dist / max_len) if max_len > 0 else 0.0


# ==========================================
# 1. 只有 Char 分数
# ==========================================
def extract_ocr_char_score(recognized_list: List[Union[str, list]], 
                           target_texts: List[Union[str, list]]) -> List[float]:
    """
    只计算字符级别的归一化编辑距离分数。
    关注拼写细节。
    """
    scores = []
    for recog, target in zip(recognized_list, target_texts):
        # 数据清洗：转为字符串
        recog_str = "".join(recog) if isinstance(recog, list) else str(recog)
        target_str = "".join(target) if isinstance(target, list) else str(target)
        
        score = _compute_single_char_score(recog_str, target_str)
        scores.append(float(score))
    return scores


def extract_ocr_word_score(recognized_list: List[Union[str, list]], 
                           target_texts: List[Union[str, list]]) -> List[float]:
    """
    只计算单词级别的归一化编辑距离分数。
    关注单词是否丢失或乱序。
    """
    scores = []
    for recog, target in zip(recognized_list, target_texts):
        # 数据清洗
        recog_str = "".join(recog) if isinstance(recog, list) else str(recog)
        target_str = "".join(target) if isinstance(target, list) else str(target)
        
        score = _compute_single_word_score(recog_str, target_str)
        scores.append(float(score))
    return scores

# ==========================================
# 3. 加权融合分数 (Weighted)
# ==========================================
def extract_ocr_weighted_score(recognized_list: List[Union[str, list]], 
                               target_texts: List[Union[str, list]], 
                               char_weight: float = 0.5, 
                               word_weight: float = 0.5) -> List[float]:
    """
    同时计算并返回加权后的综合分数。
    """
    scores = []
    
    # 归一化权重
    total_weight = char_weight + word_weight
    if total_weight <= 0:
        c_w, w_w = 0.5, 0.5
    else:
        c_w = char_weight / total_weight
        w_w = word_weight / total_weight

    for recog, target in zip(recognized_list, target_texts):
        recog_str = "".join(recog) if isinstance(recog, list) else str(recog)
        target_str = "".join(target) if isinstance(target, list) else str(target)
        
        # 分别计算
        c_score = _compute_single_char_score(recog_str, target_str)
        w_score = _compute_single_word_score(recog_str, target_str)
        
        # 加权
        final_score = (c_score * c_w) + (w_score * w_w)
        scores.append(float(final_score))
        
    return scores



import itertools

class EditDistanceMatcher:
    def __init__(self, max_n=3):
        self.max_n = max_n

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """标准的编辑距离算法 (DP实现)"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def _calculate_group_score(self, list_a: list[str], list_b: list[str]) -> tuple[float, float]:
        """
        修正版逻辑：
        1. 统一去空格：分母和分子都基于去空格后的字符串计算。
        2. 增加平局处理：当距离相同时，取更长的长度以保证分数最大化。
        """
        # 生成拼接
        perms_a = ["".join(p) for p in itertools.permutations(list_a)]
        perms_b = ["".join(p) for p in itertools.permutations(list_b)]
        
        min_dist = float('inf')
        best_max_len = 0
        
        for pa in perms_a:
            for pb in perms_b:
                # 1. 统一标准：全部去空格
                compare_a = pa.replace(" ", "")
                compare_b = pb.replace(" ", "")
                
                # 2. 计算当前对的长度 (使用去空格后的长度！)
                len_a = len(compare_a)
                len_b = len(compare_b)
                current_max_len = max(len_a, len_b)
                
                # 3. 计算距离
                dist = self._levenshtein_distance(compare_a, compare_b)
                
                # 4. 更新最优解 (核心修改)
                if dist < min_dist:
                    # 发现更小的距离，直接更新
                    min_dist = dist
                    best_max_len = current_max_len
                elif dist == min_dist:
                    # 【重要】如果距离一样，取分母更大的（这样分数更高）
                    # Score = 1 - (dist / len)，分母越大，扣分越少
                    if current_max_len > best_max_len:
                        best_max_len = current_max_len
        
        # 边界处理：防止全空字符串导致除以0
        if best_max_len == 0:
            # 如果两个都是空的，且距离为0，则是完美匹配
            if min_dist == 0:
                return 1.0, 0.0
            return 0.0, 0.0

        score = 1.0 - (min_dist / best_max_len)
        weight = score * best_max_len
        
        return score, weight
    def compute_score(self, pred_list: list[str], gt_list: list[str]) -> float:
        """
        主入口：计算两个列表的最佳匹配分数 (修复复杂度爆炸版本)
        """
        if not pred_list and not gt_list: return 1.0
        if not pred_list or not gt_list: return 0.0
        
        candidates = []
        
        # === 核心优化 1：只取相邻的行组合 (Sliding Window) ===
        def get_adjacent_subsets(length: int, max_n: int):
            subsets = []
            for r in range(1, max_n + 1):
                for i in range(length - r + 1):
                    subsets.append(tuple(range(i, i + r)))
            return subsets
            
        gt_subsets = get_adjacent_subsets(len(gt_list), self.max_n)
        pred_subsets = get_adjacent_subsets(len(pred_list), self.max_n)
        
        # 遍历生成的连续子集
        for gt_sub in gt_subsets:
            gt_strs = [gt_list[i] for i in gt_sub]
            set_g = set("".join(gt_strs)) # 提前提取字符集合
            
            for pred_sub in pred_subsets:
                pred_strs = [pred_list[i] for i in pred_sub]
                set_p = set("".join(pred_strs))
                
                # === 核心优化 2：快速过滤无交集的对 ===
                if not (set_g & set_p): 
                    continue

                # 计算这组 N-to-N 的得分
                score, weight = self._calculate_group_score(gt_strs, pred_strs)
                
                # 只保留有意义的匹配
                if score > 0.4:
                    candidates.append({
                        'gt_idxs': set(gt_sub),
                        'pred_idxs': set(pred_sub),
                        'score': score,
                        'weight': weight,
                        'debug': f"GT{list(gt_sub)} <=> Pred{list(pred_sub)}"
                    })

        # === 贪心择优 (Global Optimization) ===
        candidates.sort(key=lambda x: x['weight'], reverse=True)
        
        used_gt = set()
        used_pred = set()
        total_score_sum = 0.0
        
        for cand in candidates:
            g_ids = cand['gt_idxs']
            p_ids = cand['pred_idxs']
            
            if g_ids.isdisjoint(used_gt) and p_ids.isdisjoint(used_pred):
                used_gt.update(g_ids)
                used_pred.update(p_ids)
                total_score_sum += cand['weight']

        # 统计有效字符长度 (去空格)
        full_gt_len = sum(len(s.replace(" ", "")) for s in gt_list)
        full_pred_len = sum(len(s.replace(" ", "")) for s in pred_list)
        
        denominator = max(full_gt_len, full_pred_len)
        return total_score_sum / denominator if denominator > 0 else 0.0

    def compute_socre_all(self, pred_list: list[list[str]], gt_list: list[list[str]]) -> list[float]:
        scores = []
        for pred, gt in zip(pred_list, gt_list):
            score = self.compute_score(pred,gt)
            print("score", score)
            scores.append(score)
        return scores
        


prompt_template = '''
Your role is to evaluate the semantic alignment between the given image and the user's prompt. 
Analyze how accurately the image depicts the entities, attributes, actions, and spatial relationships described in the text.

0. Bad: Total mismatch. The image content is completely unrelated to the prompt.
1. Poor: Only vague similarities. The main subject might be present but with major contradictions or missing core elements.
2. Fair: The primary subject is correct, but there are significant errors in details such as colors, counts, or spatial arrangement.
3. Good: Highly consistent. Most elements and attributes are correctly depicted with only minor, negligible discrepancies.
4. Excellent: Perfect alignment. Every detail, attribute, and relationship mentioned in the prompt is accurately and vividly represented.

Give a final score from 0 to 4 within the <Score> tag.
<Score>X</Score>
'''

prompt_template_v2='''
Your role is to evaluate the semantic alignment between the given image and the user's prompt on a scale of 0 to 9. 
Analyze how accurately the image depicts the entities, attributes, actions, and spatial relationships described in the text.

Criteria:
0. Abysmal: Total mismatch. Image content is completely unrelated to the prompt (e.g., prompt is "a cat", image is "a car").
1. Extremely Poor: Barely related. Only the broadest category matches, but the main subject is wrong or unrecognizable.
2. Poor: The main subject is present but fundamentally incorrect (e.g., wrong species, wrong gender, or major parts missing).
3. Below Average: Subject is correct, but there are multiple major errors in attributes (colors, materials) and actions.
4. Fair/Average: Subject and primary action are correct, but significant errors exist in secondary elements or counts.
5. Good: Mostly consistent. The main scene is correct, but there are 1-2 noticeable errors in details (e.g., wrong eye color, minor object missing).
6. Above Average: High consistency. All main elements are present; only very subtle attribute or texture discrepancies exist.
7. Very Good: Very high fidelity. Captures all entities and their primary attributes. Minor issues in complex spatial relationships.
8. Excellent: Near-perfect alignment. Almost every detail, count, and relationship mentioned is accurately represented.
9. Perfect: Flawless alignment. Every single nuance, adjective, and spatial detail in the prompt is vividly and accurately depicted.

Please give a final score from 0 to 9 within the <Score> tag.

<Score>X</Score>
'''


def extract_vlm_scores(output_texts: List[str], scope=4.0) -> List[float]:
    scores = []
    for text in output_texts:
        match = re.search(r'<Score>(\d+)</Score>', text)
        if match:
            scores.append(float(match.group(1)) / scope)
        else:
            scores.append(0.0)
    return scores

    
def get_vllm_vl_reward_score(prompts: Union[str, List[str]], 
                             images: Union[torch.Tensor, List[str], List[Image.Image]], 
                             ip='6.178.129.88', port=80, token="<TOKEN>", version=1):
    prompts_list, encoded_images = pre_process(prompts, images)
    
    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
        }
    
    scores = []

    prompt_template_prifix = prompt_template
    scope = 4.0
    if version ==2:
        prompt_template_prifix = prompt_template_v2
        scope = 9.0
        
    
    for prompt, b64_image in zip(prompts_list, encoded_images):
        # Construct the OpenAI-style payload for vLLM
        payload = {
            "model": "Qwen3-VL-8B-Instruct", # Replace with your specific model name in vLLM
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_template_prifix + "\n\nUser Prompt: " + prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 256, # Adjust based on how long the "Thought" process is
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Note: vLLM returns text. If you are using this as a reward score,
            # you likely need to parse the numerical value out of the 'content'.
            content = result['choices'][0]['message']['content']
            
            # Logic to extract score from text (Example: "<Score>3</Score>")
            # This is a placeholder; adjust based on your prompt's output format
            score = extract_vlm_scores(output_texts=[content], scope=scope)[0]
            scores.append(score)

        except Exception as e:
            print(f"Error calling vLLM: {e}")
            scores.append(0.0) # Fallback value

    return torch.tensor(scores)


import re
import difflib
from typing import List

def collapse_duplicates(tokens: List[str], max_repeats: int = 2) -> List[str]:
    """
    清理连续重复的幻觉 Token。
    例如 max_repeats=2 时，'可' 连续出现 100 次，会被截断为保留 2 次。
    """
    if not tokens: return []
    
    cleaned = []
    current_token = None
    repeat_count = 0
    
    for token in tokens:
        if token == current_token:
            repeat_count += 1
            if repeat_count <= max_repeats:
                cleaned.append(token)
        else:
            current_token = token
            repeat_count = 1
            cleaned.append(token)
            
    return cleaned

def calculate_robust_reward(list_a: List[str], list_b: List[str]) -> float:
    if not list_b: return 1.0 if not list_a else 0.0
    
    # 1. 拼接并用正则打散（英文按词，中文按字）
    text_a = " ".join([str(x) for x in list_a])
    text_b = " ".join([str(x) for x in list_b])
    
    spaced_text_a = re.sub(r'([\u4e00-\u9fa5])', r' \1 ', text_a)
    spaced_text_b = re.sub(r'([\u4e00-\u9fa5])', r' \1 ', text_b)
    
    tokens_a = spaced_text_a.split()
    tokens_b = spaced_text_b.split()
    
    # 2. 对 OCR 输出 (tokens_a) 进行连续去重，防止幻觉导致计算爆炸
    # 设置 max_repeats=2 容忍像 "22" 这样的合理重复，但切断上百次的幻觉
    tokens_a = collapse_duplicates(tokens_a, max_repeats=2)
    
    len_a = len(tokens_a)
    len_b = len(tokens_b)
    
    if len_a == 0 and len_b == 0: return 1.0
    if len_a == 0 or len_b == 0: return 0.0

    # 3. 贪心匹配计算总相似度分数
    possible_matches = []
    for i, token_a in enumerate(tokens_a):
        for j, token_b in enumerate(tokens_b):
            score = difflib.SequenceMatcher(None, token_a, token_b).ratio()
            # 优化：只记录有一定相似度的匹配，减少无用计算
            if score > 0.3: 
                possible_matches.append((score, i, j))
                
    possible_matches.sort(key=lambda x: x[0], reverse=True)
    
    total_similarity = 0.0
    used_a, used_b = set(), set()
    
    for score, i, j in possible_matches:
        if i in used_a or j in used_b: continue
        total_similarity += score
        used_a.add(i)
        used_b.add(j)
        if len(used_a) == len_a or len(used_b) == len_b: break

    # 4. 计算 F1-Score (替代原本粗暴的 max_len)
    precision = total_similarity / len_a
    recall = total_similarity / len_b
    
    if precision + recall == 0:
        return 0.0
        
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def extract_ocr_score(recognized_list, target_texts: List[str|list]) -> List[float]:
    rewards: List[float] = []
    
    for recog, target in zip(recognized_list, target_texts):
        recognized_chars = set("".join(recog))  # 连接成一个字符串，方便后续匹配
        target_chars = set("".join(target))
        common_chars = recognized_chars & target_chars
        
        reward = len(common_chars) / len(target_chars) if target_chars else 0.0
        rewards.append(float(reward))
        
    return rewards


def get_vllm_ocr_reward_score(
    prompts: Union[str, List[str]],
    images: Union[List[str], List[Image.Image]],
    target_texts: Union[str, List[str]],    
    ip: str = "6.178.129.78", port: int = 80, token="<TOKEN>"
) -> torch.Tensor:
    prompts_list, encoded_images = pre_process(prompts, images)
    
    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
        }
    
    scores = []
    
    for prompt, b64_image, target in zip(prompts_list, encoded_images, target_texts):
        # print("prompt: ",prompt)
        sys_prompt = "识别并提取图片中所有的文字。请按从上到下、从左到右的阅读顺序排列。对于图片中的分栏内容，请保持分栏的逻辑结构。不要输出坐标信息，仅返回识别到的字符串。"
        # prompt=""
        payload = {
            "model": "PaddleOcr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": sys_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        },
                    ],
                }
            ],
            "temperature": 0.,
            "max_tokens": 256, # Adjust based on how long the "Thought" process is
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Note: vLLM returns text. If you are using this as a reward score,
            # you likely need to parse the numerical value out of the 'content'.
            content = result['choices'][0]['message']['content']
            content = re.split(r'\n+', content)
            score = calculate_robust_reward(content, target)
            print(f"paddle-ocr prompt: {prompt} \n content: {content}, tgt text: {target}, score: {score}")
            scores.append(score)
        except Exception as e:
            print(f"Error calling vLLM: {e}")

    # scores = extract_ocr_score(all_texts, target_texts)
    
    return scores




def get_vllm_ocr_reward_score_n2n(
    prompts: Union[str, List[str]],
    images: Union[List[str], List[Image.Image]],
    target_texts: Union[str, List[str]],    
    ip: str = "6.178.129.78", port: int = 80, token="<TOKEN>"
) -> torch.Tensor:
    prompts_list, encoded_images = pre_process(prompts, images)
    
    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
        }
    
    all_texts = []
    
    for prompt, b64_image, target in zip(prompts_list, encoded_images, target_texts):
        payload = {
            "model": "PaddleOcr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 512, # Adjust based on how long the "Thought" process is
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Note: vLLM returns text. If you are using this as a reward score,
            # you likely need to parse the numerical value out of the 'content'.
            content = result['choices'][0]['message']['content']
            content = re.split(r'\n+', content)
            print(f"content: {content}")
            all_texts.append(content)
        except Exception as e:
            print(f"Error calling vLLM: {e}")
            all_texts.append([])

    matcher = EditDistanceMatcher(max_n=3)
    
    scores = matcher.compute_socre_all(all_texts, target_texts)
    return scores





# 假设你外部有这个导入
# from .utils import pre_process 

def get_vllm_ocr_reward_score_yomi(
    prompts: Union[str, List[str]],
    images: Union[List[str], List[Image.Image]],
    target_texts: Union[str, List[str]],    
    ip: str = "6.178.129.78", port: int = 80, token="<TOKEN>"
) -> torch.Tensor:
    
    # ---------------- 内部 F1 算分辅助函数 ----------------
    def _calculate_f1_ocr_reward(ocr_text_list: List[str], tgt_text: Union[str, List[str]]) -> float:
        if isinstance(tgt_text, str):
            tgt_text = [tgt_text]
            
        if not tgt_text and not ocr_text_list: return 1.0
        if not tgt_text or not ocr_text_list: return 0.0

        tgt_lower = [str(t).lower() for t in tgt_text]
        ocr_lower = [str(o).lower() for o in ocr_text_list]
        
        # 1. 计算 Recall
        ocr_concat = " | ".join(ocr_lower)
        recall_scores = []
        for gt in tgt_lower:
            gt_clean = gt.strip()
            if not gt_clean: continue
            matcher = difflib.SequenceMatcher(None, gt_clean, ocr_concat)
            matched_length = sum(block.size for block in matcher.get_matching_blocks())
            recall_scores.append(min(1.0, matched_length / len(gt_clean)))
        recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

        # 2. 计算 Precision
        def _clean_pool(text_list):
            return re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', "".join(text_list).lower())
            
        gt_chars_str = _clean_pool(tgt_text)
        ocr_chars_str = _clean_pool(ocr_text_list)
        
        gt_counter = Counter(gt_chars_str)
        ocr_counter = Counter(ocr_chars_str)
        
        matched_chars_count = sum((gt_counter & ocr_counter).values())
        total_ocr_chars = len(ocr_chars_str)
        precision = matched_chars_count / total_ocr_chars if total_ocr_chars > 0 else 0.0

        # 3. 计算 F1
        if recall + precision == 0: return 0.0
        return 2 * (recall * precision) / (recall + precision)
    # --------------------------------------------------------

    prompts_list, encoded_images = pre_process(prompts, images)
    
    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    scores = []
    
    for prompt_txt, b64_image, target in zip(prompts_list, encoded_images, target_texts):
        # 使用强约束 Prompt，严禁模型输出多余废话，配合正则 \n+ 切分
        vl_prompt = "Accurately extract all text from the image, ignoring layout interference. Output the recognized text line by line in reading order. Note: Output ONLY raw text. Strictly do not include any conversational filler, prefixes, or extraneous punctuation."
        
        payload = {
            "model": "PaddleOcr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vl_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            # 将模型输出按行切分为列表
            content_list = [line.strip() for line in re.split(r'\n+', content) if line.strip()]
            
            # 调用全新的 F1 算法
            score = _calculate_f1_ocr_reward(content_list, target)
            print(f"paddle-ocr content: {content_list}, tgt text: {target}, score: {score:.4f}")
            scores.append(score)
            
        except Exception as e:
            print(f"Error calling vLLM/PaddleOCR: {e}")
            # 【重要修复】：发生异常时必须补充默认分数 0.0，否则会导致 batch 维度不齐报错
            scores.append(0.0)
            
    # 如果外层期待 Tensor 格式，可按需转换：
    # return torch.tensor(scores, dtype=torch.float32)
    return scores



def get_vllm_ocr_reward_score_yomi_v1(
    prompts: Union[str, List[str]],
    images: Union[List[str], List[Image.Image]],
    target_texts: Union[str, List[str]],    
    ip: str = "6.178.129.78", port: int = 80, token="<TOKEN>"
) -> torch.Tensor:
    
    # ---------------- 内部 F1 算分辅助函数 ----------------
    def _calculate_f1_ocr_reward(ocr_text_list: List[str], tgt_text: Union[str, List[str]]) -> float:
        if isinstance(tgt_text, str):
            tgt_text = [tgt_text]
            
        if not tgt_text and not ocr_text_list: return 1.0
        if not tgt_text or not ocr_text_list: return 0.0

        tgt_lower = [str(t).lower() for t in tgt_text]
        ocr_lower = [str(o).lower() for o in ocr_text_list]
        
        # 1. 计算 Recall
        ocr_concat = " | ".join(ocr_lower)
        recall_scores = []
        for gt in tgt_lower:
            gt_clean = gt.strip()
            if not gt_clean: continue
            matcher = difflib.SequenceMatcher(None, gt_clean, ocr_concat)
            matched_length = sum(block.size for block in matcher.get_matching_blocks())
            recall_scores.append(min(1.0, matched_length / len(gt_clean)))
        recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

        # 2. 计算 Precision
        def _clean_pool(text_list):
            return re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', "".join(text_list).lower())
            
        gt_chars_str = _clean_pool(tgt_text)
        ocr_chars_str = _clean_pool(ocr_text_list)
        
        gt_counter = Counter(gt_chars_str)
        ocr_counter = Counter(ocr_chars_str)
        
        matched_chars_count = sum((gt_counter & ocr_counter).values())
        total_ocr_chars = len(ocr_chars_str)
        precision = matched_chars_count / total_ocr_chars if total_ocr_chars > 0 else 0.0

        # 3. 计算 F1
        if recall + precision == 0: return 0.0
        return 2 * (recall * precision) / (recall + precision)
    # --------------------------------------------------------

    prompts_list, encoded_images = pre_process(prompts, images)
    
    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    scores = []
    
    for prompt_txt, b64_image, target in zip(prompts_list, encoded_images, target_texts):
        # 使用强约束 Prompt，严禁模型输出多余废话，配合正则 \n+ 切分
        vl_prompt = "Accurately extract all text from the image."
        
        payload = {
            "model": "PaddleOcr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vl_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            
            # --- 【新增修改：专门过滤 VLM 产生的定位标签】 ---
            # 过滤形如 <|LOC_157|>, <loc_123>, <loc> 等标签
            content = re.sub(r'<\|?loc[^>]*\|?>', '', content, flags=re.IGNORECASE)
            # 过滤形如 [loc_123] 等中括号标签
            content = re.sub(r'\[loc[^\]]*\]', '', content, flags=re.IGNORECASE)
            # --------------------------------------------------
            
            # 将模型输出按行切分为列表
            content_list = [line.strip() for line in re.split(r'\n+', content) if line.strip()]
            
            # 调用全新的 F1 算法
            score = _calculate_f1_ocr_reward(content_list, target)
            print(f"paddle-ocr content: {content_list}, tgt text: {target}, score: {score:.4f}")
            scores.append(score)
            
        except Exception as e:
            print(f"Error calling vLLM/PaddleOCR: {e}")
            # 【重要修复】：发生异常时必须补充默认分数 0.0，否则会导致 batch 维度不齐报错
            scores.append(0.0)
            
    # 如果外层期待 Tensor 格式，可按需转换：
    # return torch.tensor(scores, dtype=torch.float32)
    return scores


# def get_vllm_ocr_reward_score(
#     prompts: Union[str, List[str]],
#     images: Union[List[str], List[Image.Image]],
#     target_texts: Union[str, List[str]],    
#     ip: str = "6.178.129.78", port: int = 80, token="<TOKEN>"
# ) -> torch.Tensor:
#     prompts_list, encoded_images = pre_process(prompts, images)
    
#     url = f"http://{ip}:{port}/v1/chat/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {token}"
#         }
    
#     all_texts = []
    
#     for prompt, b64_image, target in zip(prompts_list, encoded_images, target_texts):
#         payload = {
#             "model": "PaddleOcr",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": f"data:image/png;base64,{b64_image}"}
#                         },
#                     ],
#                 }
#             ],
#             "temperature": 0,
#             "max_tokens": 256, # Adjust based on how long the "Thought" process is
#         }

#         try:
#             response = requests.post(url, headers=headers, json=payload, timeout=120)
#             response.raise_for_status()
#             result = response.json()
            
#             # Note: vLLM returns text. If you are using this as a reward score,
#             # you likely need to parse the numerical value out of the 'content'.
#             content = result['choices'][0]['message']['content']
            
#             all_texts.append([text.strip() for text in content])
#         except Exception as e:
#             print(f"Error calling vLLM: {e}")
#             all_texts.append([])

#     scores = extract_ocr_score(all_texts, target_texts)
#     return scores




def get_vllm_ocr_reward_score_word(
    prompts: Union[str, List[str]],
    images: Union[List[str], List[Image.Image]],
    target_texts: Union[str, List[str]],    
    ip: str = "6.178.129.78", port: int = 80, token="<TOKEN>"
) -> torch.Tensor:
    prompts_list, encoded_images = pre_process(prompts, images)
    
    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
        }
    
    all_texts = []
    
    for prompt, b64_image, target in zip(prompts_list, encoded_images, target_texts):
        payload = {
            "model": "PaddleOcr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 256, # Adjust based on how long the "Thought" process is
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Note: vLLM returns text. If you are using this as a reward score,
            # you likely need to parse the numerical value out of the 'content'.
            content = result['choices'][0]['message']['content']
            content = re.split(r'\n+', content)
            # print(f"content: {content}")
            all_texts.append(content)
        except Exception as e:
            print(f"Error calling vLLM: {e}")
            all_texts.append([])
    scores = extract_ocr_word_score(all_texts, target_texts)
    return scores



def get_vllm_ocr_reward_score_char(
    prompts: Union[str, List[str]],
    images: Union[List[str], List[Image.Image]],
    target_texts: Union[str, List[str]],    
    ip: str = "6.178.129.78", port: int = 80, token="<TOKEN>"
) -> torch.Tensor:
    prompts_list, encoded_images = pre_process(prompts, images)
    
    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
        }
    
    all_texts = []
    
    for prompt, b64_image, target in zip(prompts_list, encoded_images, target_texts):
        payload = {
            "model": "PaddleOcr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 256, # Adjust based on how long the "Thought" process is
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Note: vLLM returns text. If you are using this as a reward score,
            # you likely need to parse the numerical value out of the 'content'.
            content = result['choices'][0]['message']['content']
            content = re.split(r'\n+', content)
            # print(f"content: {content}")
            all_texts.append(content)
        except Exception as e:
            print(f"Error calling vLLM: {e}")
            all_texts.append([])

    scores = extract_ocr_char_score(all_texts, target_texts)
    return scores




def get_vllm_ocr_reward_score_weight_8char_2word(
    prompts: Union[str, List[str]],
    images: Union[List[str], List[Image.Image]],
    target_texts: Union[str, List[str]],    
    ip: str = "6.178.129.78", port: int = 80, token="<TOKEN>"
) -> torch.Tensor:
    prompts_list, encoded_images = pre_process(prompts, images)
    
    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
        }
    
    all_texts = []
    
    for prompt, b64_image, target in zip(prompts_list, encoded_images, target_texts):
        payload = {
            "model": "PaddleOcr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 256, # Adjust based on how long the "Thought" process is
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Note: vLLM returns text. If you are using this as a reward score,
            # you likely need to parse the numerical value out of the 'content'.
            content = result['choices'][0]['message']['content']
            content = re.split(r'\n+', content)
            # print(f"content: {content}")
            all_texts.append(content)
        except Exception as e:
            print(f"Error calling vLLM: {e}")
            all_texts.append([])

    scores = extract_ocr_weighted_score(all_texts, target_texts, char_weight=0.8, word_weight=0.2)
    return scores


# ==========================================
# 测试用例
# ==========================================
if __name__ == "__main__":
    matcher = EditDistanceMatcher(max_n=3)

    # 复杂案例：
    # 1. "人工智能" 被拆成 ["人工", "智能"] (N-to-1)
    # 2. "财务部" 错写成 "材务部" (有编辑距离)
    # 3. 顺序完全打乱
    gt = [['a'], ['12','34']]
    pred = [['a1'],['2','3','4']] 

    score = matcher.compute_socre_all(pred, gt)
    
    print(f"GT:   {gt}")
    print(f"Pred: {pred}")
    print("-" * 30)
    print(f"Final Score: {score}")

