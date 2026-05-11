#!/bin/bash
# 配置参数
HOST="0.0.0.0"
PORT=8001
WORKERS=1 # 注意：由于模型占显存，除非显存充足，否则建议先设为 1
LOG_LEVEL="info"

cd "$(dirname "$0")/../.."

ENV_BIN="${CONDA_PREFIX}/bin"
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Starting HPSv3 Reward Service on $HOST:$PORT..."

# 使用 uvicorn 启动
# --timeout 120: 防止大图推理超时
# --preload: 在主进程加载模型，节省子进程内存（部分显卡环境需关闭）
exec $ENV_BIN/gunicorn flow_grpo.server.hpsv3_scorer_multi:app \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind $HOST:$PORT \
    --timeout 120 \
    --keep-alive 5 \
    --log-level $LOG_LEVEL \
    --access-logfile -