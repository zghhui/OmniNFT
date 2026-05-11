import os
import torch
from hpsv3 import HPSv3RewardInferencer
import uvicorn
import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from typing import List
from hpsv3 import HPSv3RewardInferencer

# --- 模型封装层 ---

class HPSv3Service(HPSv3RewardInferencer):
    def __init__(self, config_path="flow_grpo/HPSv3/hpsv3/config/HPSv3_7B.yaml", checkpoint_path=os.environ.get("HPSV3_CKPT", "checkpoints/HPSv3.safetensors"), device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
        self.model = self.model.to(device)
        print(f"HPSv3 Model loaded on {device}")

    @torch.no_grad()
    def score(self, prompts, images):
        try:
            # HPSv3 内部通常会处理 PIL Image，所以直接传递
            batch = self.prepare_batch(images, prompts)
            rewards = self.model(return_dict=True, **batch)["logits"]
            print(rewards)
            return rewards.cpu().tolist()  # 转换为可序列化的 List
        except Exception as e:
            print(f" Error: {e}")
            raise e
# --- FastAPI 逻辑层 ---

app = FastAPI(title="HPSv3 Reward API")

# 全局模型实例，避免每个请求重复加载
model_service = HPSv3Service()

class RewardRequest(BaseModel):
    prompts: List[str]
    images_base64: List[str]  # 接收 base64 字符串

@app.post("/predict")
async def predict_reward(request: RewardRequest):
    if len(request.prompts) != len(request.images_base64):
        raise HTTPException(status_code=400, detail="Prompts and images count must match")

    try:
        # 解码 Base64 图像
        pil_images = []
        for img_str in request.images_base64:
            img_data = base64.b64decode(img_str)
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            pil_images.append(image)

        # 推理
        print(request.prompts)
        scores = model_service.score(request.prompts, pil_images)
        
        return {"scores": scores, "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)