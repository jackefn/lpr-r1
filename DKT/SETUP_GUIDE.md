# DKT 环境设置指南

## ✅ 已完成的工作

从 `/mnt/hpfs/xiangc/mxy/models/research/recommend/SRC` 成功复制并整理了 DKT 相关代码到独立目录。

### 📦 复制的内容

1. **核心模块** (`KTScripts/`)
   - `BackModels.py` - 基础模型（MLP, Transformer, DKT, CoKT 等）
   - `DataLoader.py` - 数据加载器（KTDataset, RecDataset, RetrievalDataset）
   - `PredictModel.py` - 模型封装（ModelWithLoss, ModelWithOptimizer）
   - `options.py` - 配置选项
   - `utils.py` - 工具函数（加载模型、评估等）

2. **训练脚本**
   - `trainDKT.py` - DKT 模型训练脚本
   - `preprocess_assist09.py` - 数据预处理脚本

3. **数据和模型**
   - `data/assist09/assist09.npz` - 预处理后的 ASSIST09 数据（8.61 MB）
   - `SavedModels/DKT_assist09.ckpt` - 训练好的 DKT 模型（19 MB）

4. **新增文件**
   - `kt_env.py` - **核心环境封装** ⭐
   - `test_dkt_env.py` - 完整测试脚本
   - `example_usage.py` - 使用示例
   - `quick_test.py` - 快速验证脚本
   - `README_DKT.md` - 详细文档

---

## 🚀 快速开始

### 1. 验证环境

```bash
cd /mnt/hpfs/xiangc/mxy/lpr-r1/DKT
conda activate lprr1
python quick_test.py
```

**预期输出：**
```
🎉 所有核心功能测试通过！

环境信息:
  - 技能数量: 35978
  - 数据集大小: 8213 个学生
  - 初始掌握度范围: [0.555, 0.563]
```

### 2. 运行完整测试

```bash
python test_dkt_env.py
```

这将运行 5 个测试：
1. ✅ 基本功能测试
2. ✅ 增量学习测试
3. ✅ 不同目标数量测试
4. ✅ 真实学生数据测试
5. ✅ 批量处理效率测试

### 3. 查看使用示例

```bash
python example_usage.py
```

包含 5 个示例场景：
1. 基本使用流程
2. 比较不同学习路径
3. 批量处理多个学生
4. 使用真实学生数据
5. 自适应学习路径

---

## 📊 数据说明

### ASSIST09 数据集

- **来源**: ASSISTments 2009-2010
- **学生数**: 8,213
- **技能/题目数**: 35,978
- **平均序列长度**: 123.05
- **最大序列长度**: 2,250
- **平均正确率**: 57.67%

### 数据格式

NPZ 文件包含三个字段：
- `skill`: 技能 ID 序列，shape: `(num_students,)`, 每个元素是变长数组
- `y`: 答题结果序列，shape: `(num_students,)`, 0/1 表示错/对
- `real_len`: 每个学生的实际序列长度，shape: `(num_students,)`

---

## 🔧 核心 API

### `KTEnv` 类

```python
from kt_env import KTEnv
import numpy as np

# 创建环境
env = KTEnv(model_name='DKT', dataset_name='assist09')

# 重置环境
targets = np.array([[100, 200, 300]])  # 目标知识点
initial_logs = np.array([[10, 20, 30, 40, 50]])  # 初始历史
state = env.reset(targets, initial_logs)

# 执行学习步骤
for step in range(10):
    kc = np.array([[101 + step]])  # 选择要学习的知识点
    step_info = env.step(kc)
    print(f"掌握度: {step_info['current_target_score'][0]:.4f}")

# 计算奖励
reward = env.get_reward(full_score=3)
print(f"最终奖励: {reward[0]:.4f}")
```

### 关键方法

| 方法 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `reset(targets, initial_logs)` | 重置环境 | 目标知识点、初始记录 | 初始状态信息 |
| `step(kc_ids, binary=True)` | 执行学习步骤 | 知识点 ID | 学习后状态 |
| `evaluate()` | 评估掌握度 | - | 当前掌握度 |
| `get_reward(full_score)` | 计算奖励 | 满分 | 归一化奖励 |
| `get_student_data(id)` | 获取学生数据 | 学生 ID | 学生学习记录 |

---

## 🎯 使用场景

### 场景 1: 强化学习环境

DKT 环境实现了标准的 RL 接口，可以直接用于训练学习路径推荐策略：

```python
# Pseudo-code
for episode in range(num_episodes):
    state = env.reset(targets, initial_logs)
    
    for step in range(max_steps):
        action = policy.select_action(state)  # LLM 或 RL Agent
        step_info = env.step(action)
        
        # 计算 reward
        reward = env.get_reward(full_score=num_targets)
        
        # 更新策略
        policy.update(state, action, reward)
```

### 场景 2: 评估学习路径质量

比较不同策略生成的学习路径：

```python
# 评估多条路径
paths = [
    generate_path_random(),
    generate_path_greedy(),
    generate_path_llm()
]

for i, path in enumerate(paths):
    env.reset(targets, initial_logs)
    env.step(path)
    reward = env.get_reward(full_score=3)
    print(f"路径 {i+1} 奖励: {reward.mean():.4f}")
```

### 场景 3: 与 Graph-R1 集成

作为 Graph-R1 的工具环境：

```python
class LearningPathTool:
    def __init__(self):
        self.env = KTEnv()
    
    def execute(self, student_id, targets, kc_to_learn):
        # 1. 从 student_id 加载历史
        student_data = self.env.get_student_data(student_id)
        initial_logs = student_data['skill_sequence'][:10]
        
        # 2. 重置环境
        self.env.reset(targets, initial_logs)
        
        # 3. 执行学习
        step_info = self.env.step([[kc_to_learn]])
        
        # 4. 返回状态（供 LLM 阅读）
        return {
            'current_mastery': step_info['current_target_score'][0],
            'targets': targets,
            'learned_kc': kc_to_learn
        }
```

---

## 📈 性能指标

### 批量处理效率

在 CPU 上的测试结果：

| Batch Size | 总耗时 | 单样本耗时 |
|------------|--------|------------|
| 1 | 2.93 秒 | 2932 ms |
| 8 | 2.99 秒 | 374 ms |
| 32 | 2.98 秒 | **93 ms** |

**建议**：使用 batch_size >= 8 以获得更好的效率。

### 模型性能

DKT 模型在 ASSIST09 上的性能（2 epochs）：

- **验证 AUC**: 82.33%
- **测试 AUC**: 82.25%
- **验证 ACC**: 75.18%
- **测试 ACC**: 75.19%

---

## 🔄 与原项目的区别

### 改进点

1. ✅ **独立封装**: `kt_env.py` 提供了清晰的 RL 接口
2. ✅ **易于使用**: 不需要了解 MindSpore 细节
3. ✅ **完整文档**: README 和示例代码
4. ✅ **测试覆盖**: 5 个测试场景确保功能正常
5. ✅ **批量优化**: 支持高效的批量处理

### 兼容性

- ✅ 保留了原始的 `trainDKT.py` 训练脚本
- ✅ 数据格式完全兼容
- ✅ 模型权重可以直接加载
- ✅ 所有原始功能都可用

---

## 🛠️ 故障排除

### 问题 1: 找不到模块

```bash
ModuleNotFoundError: No module named 'KTScripts'
```

**解决**：确保在 `/mnt/hpfs/xiangc/mxy/lpr-r1/DKT` 目录下运行脚本。

### 问题 2: 模型文件不存在

```
⚠️ 模型文件不存在: SavedModels/DKT_assist09.ckpt
```

**解决**：
1. 从源目录复制: `cp /mnt/hpfs/xiangc/mxy/models/research/recommend/SRC/SavedModels/DKT_assist09.ckpt SavedModels/`
2. 或重新训练: `python trainDKT.py -d assist09 -m DKT -c -1 --num_epochs 10`

### 问题 3: 数据文件不存在

```
FileNotFoundError: data/assist09/assist09.npz
```

**解决**：
1. 从源目录复制: `cp -r /mnt/hpfs/xiangc/mxy/models/research/recommend/SRC/data/assist09 data/`
2. 或重新预处理: 将 CSV 文件放在当前目录，运行 `python preprocess_assist09.py`

---

## 📝 下一步计划

根据你的框架设计，接下来可以：

### 阶段 1: 构建先修图（下一步）
- [ ] 为每个 KC 生成文本解释
- [ ] 使用 GraphRAG 提取先修关系
- [ ] 保存为图结构文件

### 阶段 2: 集成到 Graph-R1
- [ ] 创建 `LearningPathTool` 类
- [ ] 设计 LLM Prompt
- [ ] 实现状态→文本转换

### 阶段 3: GRPO 训练
- [ ] 定义 RL 数据格式
- [ ] 实现 reward 计算
- [ ] 启动 GRPO 训练循环

---

## 📞 支持

如有问题，请：
1. 查看 `README_DKT.md` 详细文档
2. 运行 `python quick_test.py` 验证环境
3. 查看 `example_usage.py` 中的示例

---

**状态**: ✅ 第 0 阶段完成！DKT 环境已就绪，可以开始构建学习路径推荐框架。

