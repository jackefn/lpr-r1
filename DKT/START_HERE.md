# 🚀 从这里开始

## 快速验证（30 秒）

```bash
cd /mnt/hpfs/xiangc/mxy/lpr-r1/DKT
conda activate lprr1
python quick_test.py
```

看到 `🎉 所有核心功能测试通过！` 就可以开始使用了！

---

## 最简使用示例（5 行代码）

```python
from kt_env import KTEnv
import numpy as np

env = KTEnv()  # 创建环境
env.reset(np.array([[100, 200, 300]]), np.array([[10, 20, 30]]))  # 重置
env.step(np.array([[101]]))  # 执行学习
reward = env.get_reward(full_score=3)  # 获取奖励
print(f"奖励: {reward[0]:.4f}")  # 输出结果
```

---

## 📚 文档导航

- 🏃 **快速开始** → `quick_test.py`（运行即可）
- 📖 **API 文档** → `README_DKT.md`（完整 API 说明）
- 🎓 **使用示例** → `example_usage.py`（5 个实际场景）
- 🔧 **设置指南** → `SETUP_GUIDE.md`（详细配置）
- 📊 **完整总结** → `SUMMARY.md`（所有信息）

---

## 核心文件

| 文件 | 作用 |
|------|------|
| **`kt_env.py`** | ⭐ 核心环境封装（你主要会用到这个）|
| `quick_test.py` | 快速验证（确保环境正常）|
| `test_dkt_env.py` | 完整测试（5 个测试场景）|
| `example_usage.py` | 使用示例（学习如何使用）|

---

## 下一步

### ✅ 当前状态：第 0 阶段完成

DKT 环境已就绪，可以开始构建学习路径推荐框架！

### 🎯 接下来要做什么

**阶段 1: 构建知识图谱**
1. 为知识点生成文本解释
2. 使用 GraphRAG 提取先修关系
3. 保存图结构文件

**阶段 2: 集成到 Graph-R1**
1. 创建学习路径工具
2. 设计 LLM Prompt
3. 实现状态→文本转换

**阶段 3: GRPO 训练**
1. 定义训练数据
2. 实现 reward 计算
3. 启动训练循环

---

## 🆘 遇到问题？

1. **运行 `quick_test.py` 失败**  
   → 检查是否在正确的目录和 conda 环境

2. **找不到模型文件**  
   → 查看 `SETUP_GUIDE.md` 的故障排除部分

3. **不知道如何使用**  
   → 运行 `python example_usage.py` 查看实际例子

---

**就这么简单！现在可以开始使用 DKT 环境了。** 🎉

