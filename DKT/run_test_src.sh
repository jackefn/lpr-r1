#!/bin/bash

################################################################################
# SRC模型测试脚本 - Concept级别 (138 KCs)
# 使用真实测试集评估训练好的SRC模型
################################################################################

# 激活conda环境
echo "激活conda环境: lprr1"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lprr1

# 设置工作目录
cd /mnt/hpfs/xiangc/mxy/lpr-r1/DKT

# 模型配置
MODEL_PATH="./SavedModels/SRC_assist09_kcs_path0_concept138.ckpt"
DKT_MODEL_PATH="./SavedModels/DKT_assist09_kcs_concept138.ckpt"
TEST_DATA_PATH="./data/assist09/test_set_top3.json"

# 模型参数（需要与训练时保持一致）
SKILL_NUM=138
EMBED_SIZE=128
HIDDEN_SIZE=128
DROPOUT=0.3

# 测试配置
STEPS=10                # 每个episode的学习步数
MAX_SAMPLES=-1          # 最大测试样本数（-1表示测试全部）
RAND_SEED=42

# 输出配置
OUTPUT_DIR="./TestResults"
EXP_NAME="SRC_assist09_kcs_path0_concept138"

# 设备配置
CUDA=-1                 # -1表示使用CPU，0表示使用GPU 0

################################################################################
# 开始测试
################################################################################

echo "=========================================================================="
echo "SRC模型测试 - Concept级别"
echo "=========================================================================="
echo ""
echo "模型配置:"
echo "  SRC模型路径: ${MODEL_PATH}"
echo "  DKT模型路径: ${DKT_MODEL_PATH}"
echo "  测试集路径: ${TEST_DATA_PATH}"
echo ""
echo "模型参数:"
echo "  KC数量: ${SKILL_NUM}"
echo "  Embedding维度: ${EMBED_SIZE}"
echo "  隐藏层维度: ${HIDDEN_SIZE}"
echo "  Dropout: ${DROPOUT}"
echo ""
echo "测试配置:"
echo "  学习步数: ${STEPS}"
echo "  最大测试样本: ${MAX_SAMPLES} (-1表示全部)"
echo "  随机种子: ${RAND_SEED}"
echo ""
echo "=========================================================================="
echo ""

# 检查模型文件是否存在
if [ ! -f "${MODEL_PATH}" ]; then
    echo "❌ 错误: SRC模型文件不存在: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${DKT_MODEL_PATH}" ]; then
    echo "❌ 错误: DKT模型文件不存在: ${DKT_MODEL_PATH}"
    exit 1
fi

if [ ! -f "${TEST_DATA_PATH}" ]; then
    echo "❌ 错误: 测试集文件不存在: ${TEST_DATA_PATH}"
    exit 1
fi

# 运行测试
python test_src_model.py \
    --model_path "${MODEL_PATH}" \
    --dkt_model_path "${DKT_MODEL_PATH}" \
    --test_data_path "${TEST_DATA_PATH}" \
    --skill_num ${SKILL_NUM} \
    --embed_size ${EMBED_SIZE} \
    --hidden_size ${HIDDEN_SIZE} \
    --dropout ${DROPOUT} \
    --steps ${STEPS} \
    --max_samples ${MAX_SAMPLES} \
    --rand_seed ${RAND_SEED} \
    --output_dir "${OUTPUT_DIR}" \
    --exp_name "${EXP_NAME}" \
    --save_results \
    --cuda ${CUDA}

echo ""
echo "=========================================================================="
echo "测试完成!"
echo "=========================================================================="

