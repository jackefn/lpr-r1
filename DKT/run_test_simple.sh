#!/bin/bash

# SRC模型简化版测试脚本

cd /mnt/hpfs/xiangc/mxy/lpr-r1/DKT

# 模型路径
MODEL_PATH="./SavedModels/SRC_assist09_kcs_path0_concept138.ckpt"
DKT_MODEL_PATH="./SavedModels/DKT_assist09_kcs_concept138.ckpt"

# 模型参数（与训练时保持一致）
EMBED_SIZE=128
HIDDEN_SIZE=128
DROPOUT=0.3

# 测试配置
BATCH_SIZE=128
TEST_ITERATIONS=200  # 测试200个batch
STEPS=20
NUM_TARGETS=3
NUM_INITIAL_LOGS=10
RAND_SEED=42

# 输出配置
OUTPUT_DIR="./TestResults"

# 设备配置
CUDA=-1  # -1表示使用CPU

echo "=========================================================================="
echo "开始测试SRC模型"
echo "=========================================================================="

python test_src_simple.py \
    --model_path "${MODEL_PATH}" \
    --dkt_model_path "${DKT_MODEL_PATH}" \
    --embed_size ${EMBED_SIZE} \
    --hidden_size ${HIDDEN_SIZE} \
    --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --test_iterations ${TEST_ITERATIONS} \
    --steps ${STEPS} \
    --num_targets ${NUM_TARGETS} \
    --num_initial_logs ${NUM_INITIAL_LOGS} \
    --rand_seed ${RAND_SEED} \
    --output_dir "${OUTPUT_DIR}" \
    --save_results \
    --cuda ${CUDA}

echo ""
echo "=========================================================================="
echo "测试完成！"
echo "=========================================================================="

