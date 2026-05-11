import os
import traceback
import threading
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ===== 这里改成你项目中的实际导入路径 =====
# from your_module.inference import VideoVLMRewardInference
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "videoalign"))
from inference import VideoVLMRewardInference  # 假设你把上面的类放在 inference.py

# -----------------------------
# Config
# -----------------------------
LOAD_FROM_PRETRAINED = os.getenv("VIDEOALIGN_CKPT_DIR", "./checkpoints")
LOAD_FROM_PRETRAINED_STEP = int(os.getenv("VIDEOALIGN_CKPT_STEP", "-1"))
DEVICE = os.getenv("VIDEOALIGN_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE_STR = os.getenv("VIDEOALIGN_DTYPE", "bf16")  # bf16 / fp16 / fp32

if DTYPE_STR == "bf16":
    DTYPE = torch.bfloat16
elif DTYPE_STR == "fp16":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32


# -----------------------------
# FastAPI init
# -----------------------------
app = FastAPI(title="VideoAlign Reward API", version="1.0.0")
model_lock = threading.Lock()
model_service = None


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
    rewards: List[dict]


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def startup_event():
    global model_service
    try:
        print(f"[Startup] Loading model from: {LOAD_FROM_PRETRAINED}")
        print(f"[Startup] Device={DEVICE}, Dtype={DTYPE}")
        model_service = VideoVLMRewardInference(
            load_from_pretrained=LOAD_FROM_PRETRAINED,
            load_from_pretrained_step=LOAD_FROM_PRETRAINED_STEP,
            device=DEVICE,
            dtype=DTYPE,
        )
        print("[Startup] Model loaded successfully.")
    except Exception as e:
        print("[Startup] Model loading failed:")
        traceback.print_exc()
        raise e


# -----------------------------
# APIs
# -----------------------------
@app.get("/health")
def health():
    gpu_ok = torch.cuda.is_available()
    gpu_mem = None
    if gpu_ok:
        dev_id = torch.cuda.current_device()
        gpu_mem = {
            "device": torch.cuda.get_device_name(dev_id),
            "allocated_GB": round(torch.cuda.memory_allocated(dev_id) / 1024**3, 3),
            "reserved_GB": round(torch.cuda.memory_reserved(dev_id) / 1024**3, 3),
        }
    return {"status": "healthy", "gpu": gpu_ok, "gpu_mem": gpu_mem}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global model_service

    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    if len(req.video_paths) == 0:
        raise HTTPException(status_code=400, detail="video_paths cannot be empty")
    if len(req.video_paths) != len(req.prompts):
        raise HTTPException(status_code=400, detail="video_paths and prompts length must match")
    if req.fps is not None and req.num_frames is not None:
        raise HTTPException(status_code=400, detail="fps and num_frames cannot be set at the same time")

    # 校验路径存在（可选但建议）
    for vp in req.video_paths:
        if not os.path.exists(vp):
            raise HTTPException(status_code=400, detail=f"Video path not found: {vp}")

    try:
        with model_lock:
            with torch.inference_mode():
                rewards = model_service.reward(
                    video_paths=req.video_paths,
                    prompts=req.prompts,
                    fps=req.fps,
                    num_frames=req.num_frames,
                    max_pixels=req.max_pixels,
                    use_norm=req.use_norm,
                )
        return {"status": "success", "rewards": rewards}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app_videoalign:app", host="0.0.0.0", port=8000, workers=1)