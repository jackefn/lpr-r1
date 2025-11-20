# DKT 环境模块

这个目录包含了独立的 DKT（Deep Knowledge Tracing）环境，用于学习路径推荐的强化学习框架。

## 📁 目录结构

```
DKT/
├── KTScripts/              # DKT 核心模块
│   ├── BackModels.py       # 基础模型（MLP, Transformer 等）
│   ├── DataLoader.py       # 数据加载器
│   ├── options.py          # 配置选项
│   ├── PredictModel.py     # 预测模型封装
│   └── utils.py            # 工具函数
├── data/                   # 数据目录
│   └── assist09/           # ASSIST09 数据集
│       └── assist09.npz    # 预处理后的数据
├── SavedModels/            # 训练好的模型
│   └── DKT_assist09.ckpt   # DKT 模型权重
├── trainDKT.py             # DKT 训练脚本
├── preprocess_assist09.py  # 数据预处理脚本
├── kt_env.py               # **DKT 环境封装（核心）**
├── test_dkt_env.py         # 环境测试脚本
├── requirements.txt        # Python 依赖
└── README_DKT.md           # 本文档
```

## 🚀 快速开始

### 1. 环境准备

确保已安装 MindSpore 2.x 和其他依赖：

```bash
conda activate lprr1  # 或你的环境名称
pip install -r requirements.txt
```

### 2. 测试 DKT 环境

```bash
cd /mnt/hpfs/xiangc/mxy/lpr-r1/DKT
python test_dkt_env.py
```

应该看到类似输出：
```
✅ KT 环境初始化成功
   - 模型: DKT
   - 数据集: assist09
   - 技能数量: 35978

--- 测试 1: 基本功能 ---
初始掌握度: 0.4532
最终掌握度: 0.6721
归一化奖励: 0.4015

🎉 所有测试通过！DKT 环境工作正常！
```

### 3. 使用 DKT 环境（Python 代码）

```python
from kt_env import KTEnv
import numpy as np

# 创建环境
env = KTEnv(model_name='DKT', dataset_name='assist09')

# 设置学习场景
batch_size = 4
targets = np.random.randint(0, env.skill_num, (batch_size, 3))      # 3 个目标知识点
initial_logs = np.random.randint(0, env.skill_num, (batch_size, 10))  # 10 条历史记录

# 重置环境
state_info = env.reset(targets, initial_logs)
print(f"初始掌握度: {state_info['initial_score']}")

# 执行学习路径（例如 20 步）
for step in range(20):
    kc_to_learn = np.random.randint(0, env.skill_num, (batch_size, 1))
    step_info = env.step(kc_to_learn)
    print(f"Step {step}: 目标掌握度 = {step_info['current_target_score'].mean():.4f}")

# 计算最终奖励
final_scores = env.evaluate()
rewards = env.get_reward(full_score=3)
print(f"最终奖励: {rewards.mean():.4f}")
```

## 📚 核心 API 说明

### `KTEnv` 类

#### 初始化
```python
env = KTEnv(
    model_name='DKT',        # 模型名称：'DKT' 或 'CoKT'
    dataset_name='assist09',  # 数据集名称
    data_dir='./data',        # 数据目录
    model_dir='./SavedModels' # 模型目录
)
```

#### `reset(targets, initial_logs=None)`
重置环境，开始新的学习 episode。

**参数：**
- `targets`: 目标知识点 ID，shape: `(batch_size, num_targets)`
- `initial_logs`: 学生的初始学习记录，shape: `(batch_size, seq_len)`（可选）

**返回：**
- `state_info`: 包含初始状态的字典
  ```python
  {
      'initial_score': array([...]),  # 初始掌握度
      'targets': array([...]),         # 目标知识点
      'skill_num': 35978               # 技能总数
  }
  ```

#### `step(kc_ids, binary=True)`
执行一步学习：学生学习指定的知识点。

**参数：**
- `kc_ids`: 要学习的知识点 ID，shape: `(batch_size, 1)` 或 `(batch_size, seq_len)`
- `binary`: 是否将学习结果二值化（默认 True）

**返回：**
- `step_info`: 包含学习后状态的字典
  ```python
  {
      'learning_scores': array([...]),        # 学习得分
      'current_target_score': array([...])    # 当前目标掌握度
  }
  ```

#### `evaluate()`
评估当前对目标知识点的掌握度。

**返回：**
- `scores`: 掌握度分数，shape: `(batch_size,)`

#### `get_reward(full_score=1.0)`
计算学习路径的奖励（归一化的学习增益）。

**公式：**
```
reward = (E_end - E_start) / (E_full - E_start + 1e-9)
```

**参数：**
- `full_score`: 满分（通常是目标数量）

**返回：**
- `rewards`: shape `(batch_size,)`

#### `get_student_data(student_id)`
获取指定学生的历史学习数据。

**返回：**
```python
{
    'skill_sequence': [...],   # 技能序列
    'answer_sequence': [...],  # 答题序列
    'length': 123              # 序列长度
}
```

## 🔧 如果需要重新训练 DKT

```bash
# 训练 DKT 模型
python trainDKT.py -d assist09 -m DKT -c -1 --num_epochs 10

# 模型会保存到 SavedModels/DKT_assist09.ckpt
```

## 📊 数据集信息

### ASSIST09
- **学生数**: 8,213
- **技能数**: 35,978
- **平均序列长度**: 123.05
- **数据格式**: NPZ (skill, y, real_len)

## 🎯 使用场景

### 1. 强化学习环境
DKT 环境可以作为标准的 RL 环境，用于训练学习路径推荐策略。

### 2. 学习效果评估
可以用来评估不同学习路径的效果：
```python
# 比较两条不同的学习路径
path_a = [...]  # 路径 A
path_b = [...]  # 路径 B

env.reset(targets, initial_logs)
env.step(path_a)
reward_a = env.get_reward(full_score=3)

env.reset(targets, initial_logs)
env.step(path_b)
reward_b = env.get_reward(full_score=3)

print(f"路径 A 奖励: {reward_a.mean():.4f}")
print(f"路径 B 奖励: {reward_b.mean():.4f}")
```

### 3. 学生模拟
模拟学生的学习过程，用于测试教学策略。

## 🔗 与 Graph-R1 集成

这个 DKT 环境设计为可以轻松集成到 Graph-R1 的 GRPO 框架中：

1. **作为工具环境**：将 `KTEnv` 封装为一个 Tool，供 LLM Agent 调用
2. **计算奖励**：使用 `get_reward()` 为 GRPO 提供训练信号
3. **状态表示**：将 `evaluate()` 的结果转换为文本，提供给 LLM

## 📝 注意事项

1. **批量处理**：环境支持批量处理（batch_size > 1），可以提高效率
2. **状态管理**：环境内部维护 LSTM hidden states，每次 `reset()` 会清空
3. **模型加载**：首次使用需要确保 `SavedModels/` 目录下有训练好的模型
4. **CPU/GPU**：默认使用 CPU，MindSpore 会自动选择设备

## 🐛 故障排除

### 问题：找不到模型文件
```
⚠️ 模型文件不存在: SavedModels/DKT_assist09.ckpt
```
**解决**：运行 `python trainDKT.py` 训练模型，或从源目录复制模型文件。

### 问题：MindSpore 版本不兼容
```
AttributeError: ... has no attribute '_auto_prefix'
```
**解决**：确保使用 MindSpore 2.x 版本，已修复相关兼容性问题。

### 问题：数据文件不存在
```
FileNotFoundError: data/assist09/assist09.npz
```
**解决**：运行 `python preprocess_assist09.py` 预处理数据。

## 📞 联系与支持

如有问题，请查看主项目的 README 或提交 Issue。

