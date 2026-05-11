import os
import io
import asyncio
import base64
import traceback
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from hpsv3 import HPSv3RewardInferencer


# -----------------------------
# Config
# -----------------------------
CONFIG_PATH = os.getenv("HPSV3_CONFIG_PATH", "flow_grpo/HPSv3/hpsv3/config/HPSv3_7B.yaml")
CKPT_PATH = os.getenv("HPSV3_CKPT_PATH", "checkpoints/HPSv3.safetensors")
GPU_IDS_STR = os.getenv("HPSV3_GPU_IDS", "0,1,2,3,4,5,6,7")
REPLICAS_PER_GPU = int(os.getenv("HPSV3_REPLICAS_PER_GPU", "2"))
CPU_DECODE_WORKERS = int(os.getenv("HPSV3_CPU_DECODE_WORKERS", "8"))


# -----------------------------
# 模型封装
# -----------------------------
class HPSv3Service(HPSv3RewardInferencer):
    def __init__(self, config_path, checkpoint_path, device):
        super().__init__(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
        self.model = self.model.to(device)
        self.device = device
        print(f"[Init] HPSv3 model loaded on {device}")

    @torch.no_grad()
    def score(self, prompts, images):
        batch = self.prepare_batch(images, prompts)
        rewards = self.model(return_dict=True, **batch)["logits"]
        return rewards.detach().cpu().tolist()


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="HPSv3 Reward API (Multi-GPU Concurrent)")

workers: List[Dict[str, Any]] = []
available_worker_queue: asyncio.Queue = None
decode_pool: ThreadPoolExecutor = None


class RewardRequest(BaseModel):
    prompts: List[str]
    images_base64: List[str]


def _decode_images(images_base64: List[str]) -> List[Image.Image]:
    pil_images = []
    for img_str in images_base64:
        img_data = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        pil_images.append(image)
    return pil_images


def _run_inference(worker: Dict[str, Any], prompts: List[str], pil_images: List[Image.Image]) -> List[float]:
    with torch.inference_mode():
        return worker["model"].score(prompts, pil_images)


@app.on_event("startup")
async def startup_event():
    global workers, available_worker_queue, decode_pool

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, multi-GPU mode requires CUDA.")

    gpu_ids = [int(x.strip()) for x in GPU_IDS_STR.split(",") if x.strip() != ""]
    n_cuda = torch.cuda.device_count()

    for gid in gpu_ids:
        if gid < 0 or gid >= n_cuda:
            raise RuntimeError(f"Invalid GPU ID: {gid}, total cuda devices={n_cuda}")

    available_worker_queue = asyncio.Queue()
    decode_pool = ThreadPoolExecutor(max_workers=CPU_DECODE_WORKERS, thread_name_prefix="img_decode")

    total_workers = len(gpu_ids) * REPLICAS_PER_GPU
    print(f"[Startup] Loading {REPLICAS_PER_GPU} replicas on each of {len(gpu_ids)} GPUs, total {total_workers} workers")

    wid = 0
    for gid in gpu_ids:
        device = f"cuda:{gid}"
        for replica in range(REPLICAS_PER_GPU):
            try:
                model = HPSv3Service(
                    config_path=CONFIG_PATH,
                    checkpoint_path=CKPT_PATH,
                    device=device
                )
                workers.append({
                    "id": wid,
                    "gpu_id": gid,
                    "replica": replica,
                    "device": device,
                    "model": model,
                    "lock": asyncio.Lock(),
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
async def predict_reward(request: RewardRequest):
    if len(request.prompts) != len(request.images_base64):
        raise HTTPException(status_code=400, detail="Prompts and images count must match")
    if len(request.prompts) == 0:
        raise HTTPException(status_code=400, detail="Empty request")

    t0 = time.monotonic()

    # CPU 预处理 offload 到线程池，不阻塞事件循环
    loop = asyncio.get_running_loop()
    pil_images = await loop.run_in_executor(decode_pool, _decode_images, request.images_base64)

    t_decode = time.monotonic()

    # 异步等待可用 worker，不阻塞事件循环
    try:
        wid = await asyncio.wait_for(available_worker_queue.get(), timeout=120)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="All workers are busy, retry later.")

    worker = workers[wid]
    t_acquire = time.monotonic()

    try:
        # GPU 推理 offload 到线程池，避免阻塞事件循环
        scores = await loop.run_in_executor(None, _run_inference, worker, request.prompts, pil_images)

        t_infer = time.monotonic()

        return {
            "status": "success",
            "worker_id": worker["id"],
            "gpu_id": worker["gpu_id"],
            "replica": worker["replica"],
            "device": worker["device"],
            "scores": scores,
            "timing": {
                "decode_ms": round((t_decode - t0) * 1000, 1),
                "queue_wait_ms": round((t_acquire - t_decode) * 1000, 1),
                "inference_ms": round((t_infer - t_acquire) * 1000, 1),
                "total_ms": round((t_infer - t0) * 1000, 1),
            }
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await available_worker_queue.put(wid)


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
        "gpu": torch.cuda.is_available(),
        "gpu_info": gpu_info
    }


if __name__ == "__main__":
    uvicorn.run("hpsv3_scorer_multi:app", host="0.0.0.0", port=8000, workers=1)
