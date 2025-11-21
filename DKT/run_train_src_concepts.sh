#!/bin/bash
# SRC训练脚本 - Concept级别（138个KC）
# 基于concept级别的DKT模型进行学习路径推荐训练

echo "========================================"
echo "SRC训练 - Concept级别（138个KC）"
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
echo ""

# 切换到工作目录
cd /mnt/hpfs/xiangc/mxy/lpr-r1/DKT

# 创建必要目录
mkdir -p SavedModels logs VisualResults

# 配置参数
AGENT="SRC"                    # Agent类型: SRC, DQN, MPC
DATASET="assist09_kcs"         # Concept级别数据集（138 KC）
DKT_MODEL="./SavedModels/DKT_assist09_kcs_concept138.ckpt"  # DKT模型路径
CUDA=-1                        # GPU设备ID，-1表示使用CPU
NUM_EPOCHS=10                  # 训练轮数
BATCH_SIZE=256                 # 批次大小
LR=0.001                       # 学习率
EMBED_SIZE=128                 # Embedding维度
HIDDEN_SIZE=128                # 隐藏层维度
DROPOUT=0.3                    # Dropout
PATH_TYPE=0                    # 学习路径类型 (0=随机, 1=重复目标, 2=循环目标, 3=混合)
STEPS=20                       # 每个episode的学习步数
NUM_TARGETS=3                  # 目标KC数量
NUM_INITIAL_LOGS=10            # 初始学习记录数量
TRAIN_ITERATIONS=200           # 每个epoch的训练迭代次数
TEST_ITERATIONS=200            # 测试迭代次数
RAND_SEED=42                   # 随机种子

echo "配置信息:"
echo "  Agent: $AGENT"
echo "  数据集: $DATASET"
echo "  DKT模型: $DKT_MODEL"
echo "  设备: $([ $CUDA -eq -1 ] && echo 'CPU' || echo 'GPU '$CUDA)"
echo "  训练轮数: $NUM_EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  学习率: $LR"
echo "  Embedding维度: $EMBED_SIZE"
echo "  Hidden维度: $HIDDEN_SIZE"
echo "  Dropout: $DROPOUT"
echo "  路径类型: $PATH_TYPE"
echo "  学习步数: $STEPS"
echo "  目标KC数: $NUM_TARGETS"
echo "  初始记录数: $NUM_INITIAL_LOGS"
echo "  训练迭代数: $TRAIN_ITERATIONS"
echo "  测试迭代数: $TEST_ITERATIONS"
echo "  随机种子: $RAND_SEED"
echo ""

# 检查DKT模型文件
if [ ! -f "$DKT_MODEL" ]; then
    echo "❌ 错误: DKT模型文件不存在: $DKT_MODEL"
    echo "请先训练DKT模型:"
    echo "  ./run_train_concept_dkt.sh"
    exit 1
fi

echo "✅ DKT模型文件检查通过"
echo ""

# 生成日志文件名
LOG_FILE="logs/train_src_concepts_$(date +%Y%m%d_%H%M%S).log"

echo "开始训练..."
echo "日志文件: $LOG_FILE"
echo ""

# 训练命令
python trainSRC_concepts.py \
    --agent $AGENT \
    --model DKT \
    --dataset $DATASET \
    --dkt_model_path $DKT_MODEL \
    --cuda $CUDA \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --embed_size $EMBED_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --dropout $DROPOUT \
    --path $PATH_TYPE \
    --steps $STEPS \
    --num_targets $NUM_TARGETS \
    --num_initial_logs $NUM_INITIAL_LOGS \
    --train_iterations $TRAIN_ITERATIONS \
    --test_iterations $TEST_ITERATIONS \
    --rand_seed $RAND_SEED \
    2>&1 | tee $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "✅ 训练完成!"
    echo "========================================"
    echo "模型保存位置: SavedModels/${AGENT}_${DATASET}_path${PATH_TYPE}_concept138.ckpt"
    echo "训练曲线: VisualResults/${AGENT}_${DATASET}_path${PATH_TYPE}_concept138_rewards.npy"
    echo "日志文件: $LOG_FILE"
else
    echo "========================================"
    echo "❌ 训练失败 (退出码: $EXIT_CODE)"
    echo "========================================"
    echo "请查看日志文件: $LOG_FILE"
fi
echo ""

exit $EXIT_CODE

