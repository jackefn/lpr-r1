#!/bin/bash
# DKT训练脚本 - Concept级别（138个KC）
# 使用lprr1 conda环境

echo "========================================"
echo "DKT训练 - Concept级别（138个KC）"
echo "========================================"

# 激活lprr1 conda环境
echo "激活conda环境: lprr1"
source /home/xiangc/anaconda3/etc/profile.d/conda.sh
conda activate lprr1

# 验证环境
echo ""
echo "环境验证:"
which python
python --version
python -c "import mindspore; print('MindSpore版本:', mindspore.__version__)" 2>/dev/null || echo "⚠️  MindSpore未安装"
echo ""

# 切换到工作目录
cd /mnt/hpfs/xiangc/mxy/lpr-r1/DKT

# 创建logs目录
mkdir -p logs

# 基本配置
MODEL="DKT"
DATASET="assist09_kcs"
CUDA=-1  # GPU设备ID，-1表示使用CPU
NUM_EPOCHS=10
BATCH_SIZE=256
LR=0.001
EMBED_SIZE=128
HIDDEN_SIZE=128
DROPOUT=0.3
L2_REG=0.00001
RAND_SEED=42

echo "配置信息:"
echo "  模型: $MODEL"
echo "  数据集: $DATASET"
echo "  GPU: $CUDA"
echo "  训练轮数: $NUM_EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  学习率: $LR"
echo "  Embedding维度: $EMBED_SIZE"
echo "  Hidden维度: $HIDDEN_SIZE"
echo "  Dropout: $DROPOUT"
echo "  L2正则化: $L2_REG"
echo "  随机种子: $RAND_SEED"
echo ""

# 检查数据文件
if [ ! -f "data/assist09_kcs/assist09_kcs.npz" ]; then
    echo "❌ 错误: 数据文件不存在 data/assist09_kcs/assist09_kcs.npz"
    exit 1
fi

echo "✅ 数据文件检查通过"
echo ""

# 生成日志文件名
LOG_FILE="logs/train_concept_dkt_$(date +%Y%m%d_%H%M%S).log"

echo "开始训练..."
echo "日志文件: $LOG_FILE"
echo ""

# 训练命令
python train_dkt_concepts.py \
    --model $MODEL \
    --dataset $DATASET \
    --cuda $CUDA \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --embed_size $EMBED_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --dropout $DROPOUT \
    --l2_reg $L2_REG \
    --rand_seed $RAND_SEED \
    2>&1 | tee $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "✅ 训练完成!"
    echo "========================================"
    echo "模型保存位置: SavedModels/DKT_assist09_kcs_concept138.ckpt"
    echo "日志文件: $LOG_FILE"
else
    echo "========================================"
    echo "❌ 训练失败 (退出码: $EXIT_CODE)"
    echo "========================================"
    echo "请查看日志文件: $LOG_FILE"
fi
echo ""

exit $EXIT_CODE

