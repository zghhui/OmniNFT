import os
import asyncio
import traceback
import time
import re
from concurrent.futures import ThreadPoolExecutor
from string import Template
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from qwen_vl_utils import process_vision_info


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = os.getenv("VS2_MODEL_NAME", "TIGER-Lab/VideoScore2")
GPU_IDS_STR = os.getenv("VS2_GPU_IDS", "0,1,2,3,4,5,6,7")
REPLICAS_PER_GPU = int(os.getenv("VS2_REPLICAS_PER_GPU", "1"))
CPU_IO_WORKERS = int(os.getenv("VS2_CPU_IO_WORKERS", "8"))
INFER_FPS = float(os.getenv("VS2_INFER_FPS", "2.0"))
MAX_NEW_TOKENS = int(os.getenv("VS2_MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("VS2_TEMPERATURE", "0.7"))

VS2_QUERY_TEMPLATE = Template("""
You are an expert for evaluating AI-generated videos from three dimensions:
(1) visual quality – clarity, smoothness, artifacts;
(2) text-to-video alignment – fidelity to the prompt;
(3) physical/common-sense consistency – naturalness and physics plausibility.

Video prompt: $t2v_prompt

Please output in this format:
visual quality: <v_score>;
text-to-video alignment: <t_score>,
physical/common-sense consistency: <p_score>
""")


# -----------------------------
# Score parsing utils
# -----------------------------
def _ll_based_soft_score_normed(hard_val, token_idx, scores, tokenizer):
    if hard_val is None or token_idx < 0:
        return None
    logits = scores[token_idx][0]
    score_probs = []
    for s in range(1, 6):
        ids = tokenizer.encode(str(s), add_special_tokens=False)
        if len(ids) == 1:
            logp = torch.log_softmax(logits, dim=-1)[ids[0]].item()
            score_probs.append((s, float(np.exp(logp))))
    if not score_probs:
        return None
    scores_list, probs_list = zip(*score_probs)
    total_prob = sum(probs_list)
    max_prob = max(probs_list)
    best_score = scores_list[probs_list.index(max_prob)]
    normalized_prob = max_prob / total_prob if total_prob > 0 else 0
    return round(best_score * normalized_prob, 4)


def _find_score_token_index_by_prompt(prompt_text, tokenizer, gen_ids):
    gen_str = tokenizer.decode(gen_ids, skip_special_tokens=False)
    pattern = r"(?:\(\d+\)\s*|\n\s*)?" + re.escape(prompt_text)
    match = re.search(pattern, gen_str, flags=re.IGNORECASE)
    if not match:
        return -1
    after_text = gen_str[match.end():]
    num_match = re.search(r"\d", after_text)
    if not num_match:
        return -1
    target_substr = gen_str[:match.end() + num_match.start() + 1]
    for i in range(len(gen_ids)):
        partial = tokenizer.decode(gen_ids[:i + 1], skip_special_tokens=False)
        if partial == target_substr:
            return i
    return -1


def _parse_scores(output_text, gen_token_ids, scores, tokenizer):
    pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
    match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
    v_hard = int(match.group(1)) if match else None
    t_hard = int(match.group(2)) if match else None
    p_hard = int(match.group(3)) if match else None

    idx_v = _find_score_token_index_by_prompt("visual quality:", tokenizer, gen_token_ids)
    idx_t = _find_score_token_index_by_prompt("text-to-video alignment:", tokenizer, gen_token_ids)
    idx_p = _find_score_token_index_by_prompt("physical/common-sense consistency:", tokenizer, gen_token_ids)

    return {
        "visual_quality": _ll_based_soft_score_normed(v_hard, idx_v, scores, tokenizer),
        "text_to_video_alignment": _ll_based_soft_score_normed(t_hard, idx_t, scores, tokenizer),
        "physical_consistency": _ll_based_soft_score_normed(p_hard, idx_p, scores, tokenizer),
        "visual_quality_hard": v_hard,
        "text_to_video_alignment_hard": t_hard,
        "physical_consistency_hard": p_hard,
        "raw_output": output_text,
    }


# -----------------------------
# Worker 封装
# -----------------------------
class VideoScore2Worker:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        print(f"[Init] VideoScore2 loaded on {device}")

    @torch.no_grad()
    def score(self, video_path: str, prompt: str, fps: float, max_new_tokens: int, temperature: float) -> dict:
        user_prompt = VS2_QUERY_TEMPLATE.substitute(t2v_prompt=prompt)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps},
            {"type": "text", "text": user_prompt}
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            fps=fps, padding=True, return_tensors="pt"
        ).to(self.device)

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=temperature,
        )

        sequences = gen_out.sequences
        scores = gen_out.scores
        input_len = inputs["input_ids"].shape[1]
        gen_token_ids = sequences[0, input_len:].tolist()

        output_text = self.processor.batch_decode(
            sequences[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return _parse_scores(output_text, gen_token_ids, scores, self.tokenizer)


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="VideoScore2 Reward API (Multi-GPU Concurrent)")

workers: List[Dict[str, Any]] = []
available_worker_queue: asyncio.Queue = None
io_pool: ThreadPoolExecutor = None


class PredictRequest(BaseModel):
    video_paths: List[str] = Field(..., description="本机可访问的视频路径列表")
    prompts: List[str] = Field(..., description="与视频一一对应的文本 prompt")
    fps: Optional[float] = Field(None, description="采样 fps，默认使用 VS2_INFER_FPS")
    max_new_tokens: Optional[int] = Field(None, description="最大生成 token 数")
    temperature: Optional[float] = Field(None, description="采样温度")


def _validate_video_paths(video_paths: List[str]):
    for vp in video_paths:
        if not os.path.exists(vp):
            raise FileNotFoundError(f"Video path not found: {vp}")


def _run_inference(worker, video_path, prompt, fps, max_new_tokens, temperature):
    with torch.inference_mode():
        return worker["model"].score(video_path, prompt, fps, max_new_tokens, temperature)


@app.on_event("startup")
async def startup_event():
    global workers, available_worker_queue, io_pool

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, multi-GPU mode requires CUDA.")

    gpu_ids = [int(x.strip()) for x in GPU_IDS_STR.split(",") if x.strip() != ""]
    n_cuda = torch.cuda.device_count()

    for gid in gpu_ids:
        if gid < 0 or gid >= n_cuda:
            raise RuntimeError(f"Invalid GPU ID: {gid}, total cuda devices={n_cuda}")

    available_worker_queue = asyncio.Queue()
    io_pool = ThreadPoolExecutor(max_workers=CPU_IO_WORKERS, thread_name_prefix="video_io")

    total_workers = len(gpu_ids) * REPLICAS_PER_GPU
    print(f"[Startup] Loading {REPLICAS_PER_GPU} replicas on each of {len(gpu_ids)} GPUs, total {total_workers} workers")
    print(f"[Startup] Model={MODEL_NAME}, FPS={INFER_FPS}")

    wid = 0
    for gid in gpu_ids:
        device = f"cuda:{gid}"
        for replica in range(REPLICAS_PER_GPU):
            try:
                model = VideoScore2Worker(model_name=MODEL_NAME, device=device)
                workers.append({
                    "id": wid,
                    "gpu_id": gid,
                    "replica": replica,
                    "device": device,
                    "model": model,
                })
                await available_worker_queue.put(wid)
                print(f"[Startup] Worker {wid} (GPU {gid}, replica {replica}) ready")
                wid += 1
            except Exception:
                print(f"[Startup] Failed on {device} replica {replica}")
                traceback.print_exc()
                raise

    print(f"[Startup] All {len(workers)} workers ready")


@app.post("/predict")
async def predict(req: PredictRequest):
    if len(req.video_paths) == 0:
        raise HTTPException(status_code=400, detail="video_paths cannot be empty")
    if len(req.video_paths) != len(req.prompts):
        raise HTTPException(status_code=400, detail="video_paths and prompts length must match")

    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    try:
        await loop.run_in_executor(io_pool, _validate_video_paths, req.video_paths)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    fps = req.fps or INFER_FPS
    max_tokens = req.max_new_tokens or MAX_NEW_TOKENS
    temp = req.temperature or TEMPERATURE

    results = []
    for video_path, prompt in zip(req.video_paths, req.prompts):
        t_wait_start = time.monotonic()

        try:
            wid = await asyncio.wait_for(available_worker_queue.get(), timeout=300)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="All workers are busy, retry later.")

        worker = workers[wid]
        t_acquire = time.monotonic()

        try:
            score_result = await loop.run_in_executor(
                None, _run_inference, worker, video_path, prompt, fps, max_tokens, temp
            )
            t_infer = time.monotonic()

            results.append({
                "video_path": video_path,
                "worker_id": worker["id"],
                "gpu_id": worker["gpu_id"],
                "replica": worker["replica"],
                **score_result,
                "timing": {
                    "queue_wait_ms": round((t_acquire - t_wait_start) * 1000, 1),
                    "inference_ms": round((t_infer - t_acquire) * 1000, 1),
                }
            })
        except Exception as e:
            traceback.print_exc()
            results.append({
                "video_path": video_path,
                "error": str(e),
            })
        finally:
            await available_worker_queue.put(wid)

    t_total = time.monotonic()

    return {
        "status": "success",
        "rewards": results,
        "timing": {
            "total_ms": round((t_total - t0) * 1000, 1),
        }
    }


@app.get("/health")
async def health_check():
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "gpu_id": i,
                "name": torch.cuda.get_device_name(i),
                "allocated_GB": round(torch.cuda.memory_allocated(i) / 1024**3, 3),
                "reserved_GB": round(torch.cuda.memory_reserved(i) / 1024**3, 3),
            })

    return {
        "status": "healthy",
        "num_workers": len(workers),
        "num_available_workers": available_worker_queue.qsize() if available_worker_queue else 0,
        "replicas_per_gpu": REPLICAS_PER_GPU,
        "model": MODEL_NAME,
        "gpu": torch.cuda.is_available(),
        "gpu_info": gpu_info,
    }


if __name__ == "__main__":
    uvicorn.run("videoscore2_scorer_multi:app", host="0.0.0.0", port=8000, workers=1)
