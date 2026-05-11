import os
import asyncio
import traceback
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "videoalign"))
from inference import VideoVLMRewardInference


# -----------------------------
# Config
# -----------------------------
LOAD_FROM_PRETRAINED = os.getenv("VIDEOALIGN_CKPT_DIR", "./checkpoints")
LOAD_FROM_PRETRAINED_STEP = int(os.getenv("VIDEOALIGN_CKPT_STEP", "-1"))
DTYPE_STR = os.getenv("VIDEOALIGN_DTYPE", "bf16")  # bf16 / fp16 / fp32
GPU_IDS_STR = os.getenv("VIDEOALIGN_GPU_IDS", "0,1,2,3,4,5,6,7")
REPLICAS_PER_GPU = int(os.getenv("VIDEOALIGN_REPLICAS_PER_GPU", "2"))
CPU_IO_WORKERS = int(os.getenv("VIDEOALIGN_CPU_IO_WORKERS", "8"))

if DTYPE_STR == "bf16":
    DTYPE = torch.bfloat16
elif DTYPE_STR == "fp16":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32


# -----------------------------
# FastAPI init
# -----------------------------
app = FastAPI(title="VideoAlign Reward API", version="2.0.0")

workers: List[Dict[str, Any]] = []
available_worker_queue: asyncio.Queue = None
io_pool: ThreadPoolExecutor = None


# -----------------------------
# Request / Response
# -----------------------------
class PredictRequest(BaseModel):
    video_paths: List[str] = Field(..., description="本机可访问的视频路径列表")
    prompts: List[str] = Field(..., description="与视频一一对应的文本prompt")
    fps: Optional[float] = Field(None, description="采样fps；与num_frames二选一")
    num_frames: Optional[int] = Field(None, description="采样帧数；与fps二选一")
    max_pixels: Optional[int] = Field(None, description="每帧最大像素")
    use_norm: bool = Field(True, description="是否做归一化")


class PredictResponse(BaseModel):
    status: str
    worker_id: int
    gpu_id: int
    replica: int
    device: str
    rewards: List[dict]
    timing: Optional[dict] = None


# -----------------------------
# Helpers
# -----------------------------
def _validate_video_paths(video_paths: List[str]):
    for vp in video_paths:
        if not os.path.exists(vp):
            raise FileNotFoundError(f"Video path not found: {vp}")


def _run_inference(worker, video_paths, prompts, fps, num_frames, max_pixels, use_norm):
    with torch.inference_mode():
        return worker["model"].reward(
            video_paths=video_paths,
            prompts=prompts,
            fps=fps,
            num_frames=num_frames,
            max_pixels=max_pixels,
            use_norm=use_norm,
        )


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global workers, available_worker_queue, io_pool

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, but this deployment expects multi-GPU.")

    gpu_ids = [int(x.strip()) for x in GPU_IDS_STR.split(",") if x.strip() != ""]
    n_cuda = torch.cuda.device_count()
    for gid in gpu_ids:
        if gid < 0 or gid >= n_cuda:
            raise RuntimeError(f"Invalid GPU id {gid}, total cuda devices = {n_cuda}")

    available_worker_queue = asyncio.Queue()
    io_pool = ThreadPoolExecutor(max_workers=CPU_IO_WORKERS, thread_name_prefix="video_io")

    total_workers = len(gpu_ids) * REPLICAS_PER_GPU
    print(f"[Startup] Loading {REPLICAS_PER_GPU} replicas on each of {len(gpu_ids)} GPUs, total {total_workers} workers")
    print(f"[Startup] Dtype={DTYPE}, GPUs={gpu_ids}")

    wid = 0
    for gid in gpu_ids:
        device = f"cuda:{gid}"
        for replica in range(REPLICAS_PER_GPU):
            try:
                model = VideoVLMRewardInference(
                    load_from_pretrained=LOAD_FROM_PRETRAINED,
                    load_from_pretrained_step=LOAD_FROM_PRETRAINED_STEP,
                    device=device,
                    dtype=DTYPE,
                )
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


# -----------------------------
# APIs
# -----------------------------
@app.get("/health")
async def health():
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "gpu_id": i,
                "device": torch.cuda.get_device_name(i),
                "allocated_GB": round(torch.cuda.memory_allocated(i) / 1024**3, 3),
                "reserved_GB": round(torch.cuda.memory_reserved(i) / 1024**3, 3),
            })
    return {
        "status": "healthy",
        "num_workers": len(workers),
        "num_available_workers": available_worker_queue.qsize() if available_worker_queue else 0,
        "replicas_per_gpu": REPLICAS_PER_GPU,
        "gpu_info": gpu_info,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if len(workers) == 0:
        raise HTTPException(status_code=503, detail="Model workers not ready")
    if len(req.video_paths) == 0:
        raise HTTPException(status_code=400, detail="video_paths cannot be empty")
    if len(req.video_paths) != len(req.prompts):
        raise HTTPException(status_code=400, detail="video_paths and prompts length must match")
    if req.fps is not None and req.num_frames is not None:
        raise HTTPException(status_code=400, detail="fps and num_frames cannot be set at the same time")

    t0 = time.monotonic()

    loop = asyncio.get_running_loop()

    # 视频路径校验 offload 到线程池（涉及文件系统 IO）
    try:
        await loop.run_in_executor(io_pool, _validate_video_paths, req.video_paths)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    t_validate = time.monotonic()

    # 异步等待可用 worker，不阻塞事件循环
    try:
        wid = await asyncio.wait_for(available_worker_queue.get(), timeout=120)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="All model workers are busy, please retry later.")

    worker = workers[wid]
    t_acquire = time.monotonic()

    try:
        # GPU 推理 offload 到线程池
        rewards = await loop.run_in_executor(
            None, _run_inference, worker,
            req.video_paths, req.prompts, req.fps, req.num_frames, req.max_pixels, req.use_norm,
        )

        t_infer = time.monotonic()

        return {
            "status": "success",
            "worker_id": worker["id"],
            "gpu_id": worker["gpu_id"],
            "replica": worker["replica"],
            "device": worker["device"],
            "rewards": rewards,
            "timing": {
                "validate_ms": round((t_validate - t0) * 1000, 1),
                "queue_wait_ms": round((t_acquire - t_validate) * 1000, 1),
                "inference_ms": round((t_infer - t_acquire) * 1000, 1),
                "total_ms": round((t_infer - t0) * 1000, 1),
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await available_worker_queue.put(wid)


if __name__ == "__main__":
    uvicorn.run("videoalign_scorer_multi:app", host="0.0.0.0", port=8000, workers=1)
