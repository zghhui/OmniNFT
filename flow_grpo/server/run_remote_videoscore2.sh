#!/bin/bash
set -e

# -------------------------
# 基础配置
# -------------------------
HOST="0.0.0.0"
PORT=8002
WORKERS=1              # 大模型建议 1，避免多进程重复占用显存
LOG_LEVEL="info"
TIMEOUT=600            # VideoScore2 generate 较慢，给足超时
KEEP_ALIVE=5

cd "$(dirname "$0")/../.."

ENV_BIN="${CONDA_PREFIX}/bin"
export PYTHONPATH=$(pwd):$PYTHONPATH

APP_MODULE="flow_grpo.server.videoscore2_scorer_multi:app"

# -------------------------
# VideoScore2 模型相关环境变量
# -------------------------
export VS2_MODEL_NAME="${VS2_MODEL_NAME:-checkpoints/VideoScore2}"
export VS2_GPU_IDS="0,1,2,3,4,5,6,7"
export VS2_REPLICAS_PER_GPU="1"
export VS2_CPU_IO_WORKERS="8"
export VS2_INFER_FPS="2.0"
export VS2_MAX_NEW_TOKENS="1024"
export VS2_TEMPERATURE="0.7"

echo "Starting VideoScore2 Reward Service on ${HOST}:${PORT} ..."
echo "App: ${APP_MODULE}"
echo "Model: ${VS2_MODEL_NAME}"
echo "GPUs: ${VS2_GPU_IDS}, Replicas/GPU: ${VS2_REPLICAS_PER_GPU}"

exec ${ENV_BIN}/gunicorn ${APP_MODULE} \
    --workers ${WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind ${HOST}:${PORT} \
    --timeout ${TIMEOUT} \
    --keep-alive ${KEEP_ALIVE} \
    --log-level ${LOG_LEVEL} \
    --access-logfile -
