#!/bin/bash

cd OmniNFT/

export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export NCCL_DEBUG=WARN

conda activate omninft

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

TOTAL_PROCESSES=$((NNODES * GPUS_PER_NODE))

echo "Running torchrun with:"
echo "  Nodes: $NNODES"
echo "  Rank: $NODE_RANK"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  Total Processes: $TOTAL_PROCESSES"

export HPSV3_REWARD_SERVER='10.185.127.97'
export HPSV3_REWARD_PORT="8001"

export VIDEOALIGN_REWARD_SERVER=$HPSV3_REWARD_SERVER
export VIDEOALIGN_REWARD_PORT="8002"

CONFIG_ITEM=${1:-"ltx_mllm_debug"}
echo "当前使用的配置项是: $CONFIG_ITEM"

torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    scripts/train_omninft_ltx_fsdp.py --config config/nft.py:$CONFIG_ITEM

sleep 1d