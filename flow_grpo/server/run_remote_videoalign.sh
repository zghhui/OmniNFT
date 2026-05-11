#!/bin/bash
set -e

# -------------------------
# 基础配置
# -------------------------
HOST="0.0.0.0"
PORT=8002
WORKERS=1              # 大模型建议 1，避免多进程重复占用显存
LOG_LEVEL="info"
TIMEOUT=300            # 视频推理通常更慢，建议比图片模型更大
KEEP_ALIVE=5


# 你的 Python 环境
cd "$(dirname "$0")/../.."

ENV_BIN="${CONDA_PREFIX}/bin"
export PYTHONPATH=$(pwd):$PYTHONPATH

# Gunicorn app 路径（按你的项目结构改）
# 假设你部署文件是 flow_grpo/videoalign_scorer.py 且里边有 app=FastAPI(...)
APP_MODULE="flow_grpo.server.videoalign_scorer_multi:app"

# -------------------------
# VideoAlign 模型相关环境变量
# -------------------------
export VIDEOALIGN_CKPT_DIR="${VIDEOALIGN_CKPT_DIR:-checkpoints/VideoReward}"
export VIDEOALIGN_CKPT_STEP="-1"
export VIDEOALIGN_DEVICE="cuda:0"
export VIDEOALIGN_DTYPE="bf16"   # bf16 / fp16 / fp32

echo "Starting VideoAlign Reward Service on ${HOST}:${PORT} ..."
echo "App: ${APP_MODULE}"
echo "CKPT: ${VIDEOALIGN_CKPT_DIR}"
echo "Device: ${VIDEOALIGN_DEVICE}, Dtype: ${VIDEOALIGN_DTYPE}"

exec ${ENV_BIN}/gunicorn ${APP_MODULE} \
    --workers ${WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind ${HOST}:${PORT} \
    --timeout ${TIMEOUT} \
    --keep-alive ${KEEP_ALIVE} \
    --log-level ${LOG_LEVEL} \
    --access-logfile -