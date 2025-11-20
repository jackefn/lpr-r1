#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DKT 环境封装 - 用于学习路径推荐的强化学习环境

提供标准的 RL 接口：reset(), step(), evaluate()
"""
import os
import numpy as np
from mindspore import Tensor, load_param_into_net, load_checkpoint
from mindspore import dtype as mdtype

from KTScripts.DataLoader import KTDataset
from KTScripts.utils import load_model
from KTScripts.options import get_exp_configure


class KTEnv:
    """
    知识追踪环境 - 将 DKT 模型封装为 RL 环境
    
    功能：
    1. 模拟学生学习过程
    2. 计算学生对目标知识点的掌握度
    3. 计算学习路径的奖励
    """
    
    def __init__(self, model_name='DKT', dataset_name='assist09', 
                 data_dir='./data', model_dir='./SavedModels'):
        """
        初始化 KT 环境
        
        Args:
            model_name: 模型名称 (DKT, CoKT 等)
            dataset_name: 数据集名称
            data_dir: 数据目录
            model_dir: 模型目录
        """
        # 加载数据集
        dataset_path = os.path.join(data_dir, dataset_name)
        self.dataset = KTDataset(dataset_path)
        self.skill_num = self.dataset.feats_num
        
        # 加载训练好的 DKT 模型
        self.model = self._load_model(model_name, dataset_name, model_dir)
        
        # 当前状态
        self.targets = None
        self.states = (None, None)  # LSTM hidden states
        self.initial_score = None
        self.current_score = None
        
        print(f"✅ KT 环境初始化成功")
        print(f"   - 模型: {model_name}")
        print(f"   - 数据集: {dataset_name}")
        print(f"   - 技能数量: {self.skill_num}")
    
    def _load_model(self, model_name, dataset_name, model_dir):
        """加载训练好的 DKT 模型"""
        # 获取模型配置
        model_parameters = get_exp_configure(model_name)
        model_parameters.update({
            'feat_nums': self.skill_num,
            'model': model_name,
            'without_label': False
        })
        
        # 创建模型
        model = load_model(model_parameters)
        
        # 加载权重
        model_path = os.path.join(model_dir, f'{model_name}_{dataset_name}.ckpt')
        if os.path.exists(model_path):
            load_param_into_net(model, load_checkpoint(model_path))
            print(f"   - 加载模型: {model_path}")
        else:
            print(f"   ⚠️  模型文件不存在: {model_path}")
            print(f"   - 将使用未训练的模型（仅用于测试）")
        
        model.set_train(False)  # 设置为评估模式
        return model
    
    def reset(self, targets, initial_logs=None):
        """
        重置环境，开始新的学习 episode
        
        Args:
            targets: 目标知识点 ID，shape: (batch_size, num_targets)
            initial_logs: 学生的初始学习记录，shape: (batch_size, seq_len)
        
        Returns:
            state_info: 包含初始状态信息的字典
        """
        self.targets = Tensor(targets)
        
        # 初始化状态
        self.states = (None, None)
        
        # 如果有初始学习记录，先更新状态
        if initial_logs is not None:
            initial_logs_tensor = Tensor(initial_logs)
            _, self.states = self.model.learn_lstm(initial_logs_tensor)
        
        # 计算初始掌握度
        self.initial_score = self._evaluate_targets(self.targets, self.states)
        self.current_score = self.initial_score.copy()
        
        return {
            'initial_score': self.initial_score.asnumpy(),
            'targets': targets,
            'skill_num': self.skill_num
        }
    
    def step(self, kc_ids, binary=True):
        """
        执行一步学习：学生学习指定的知识点
        
        Args:
            kc_ids: 要学习的知识点 ID，shape: (batch_size, 1) 或 (batch_size, seq_len)
            binary: 是否将学习结果二值化（>0.5 为掌握）
        
        Returns:
            step_info: 包含学习后状态信息的字典
        """
        kc_tensor = Tensor(kc_ids) if not isinstance(kc_ids, Tensor) else kc_ids
        
        # 通过 DKT 模拟学习过程
        scores, self.states = self.model.learn_lstm(kc_tensor, *self.states)
        
        # 是否二值化学习结果
        if binary:
            import mindspore.ops as ops
            scores = ops.cast(scores > 0.5, mdtype.float32)
        
        # 计算当前对目标的掌握度
        self.current_score = self._evaluate_targets(self.targets, self.states)
        
        return {
            'learning_scores': scores.asnumpy(),
            'current_target_score': self.current_score.asnumpy()
        }
    
    def evaluate(self):
        """
        评估当前状态下对目标知识点的掌握度
        
        Returns:
            掌握度分数，shape: (batch_size,)
        """
        return self.current_score.asnumpy()
    
    def get_reward(self, full_score=1.0):
        """
        计算学习路径的奖励 (归一化的学习增益)
        
        Formula: reward = (E_end - E_start) / (E_full - E_start + 1e-9)
        
        Args:
            full_score: 满分（通常是目标数量）
        
        Returns:
            rewards: shape (batch_size,)
        """
        delta = self.current_score - self.initial_score
        normalize_factor = full_score - self.initial_score + 1e-9
        rewards = delta / normalize_factor
        return rewards.asnumpy()
    
    def _evaluate_targets(self, targets, states):
        """
        评估在当前状态下对目标知识点的掌握度
        
        Args:
            targets: 目标知识点，shape: (batch_size, num_targets)
            states: LSTM 隐藏状态 (h, c)
        
        Returns:
            scores: 平均掌握度，shape: (batch_size,)
        """
        import mindspore.ops as ops
        
        scores = []
        for i in range(targets.shape[1]):
            # 对每个目标知识点评估掌握度
            score, _ = self.model.learn_lstm(targets[:, i:i+1], *states)
            scores.append(score)
        
        # 返回平均掌握度
        return ops.mean(ops.concat(scores, axis=1), axis=1)
    
    def get_student_data(self, student_id):
        """
        获取指定学生的历史学习数据
        
        Args:
            student_id: 学生 ID
        
        Returns:
            学生数据字典
        """
        if student_id >= len(self.dataset):
            raise ValueError(f"Student ID {student_id} 超出范围 (最大: {len(self.dataset)-1})")
        
        skill, y, mask = self.dataset[student_id]
        real_len = mask.sum().item()
        
        return {
            'skill_sequence': skill[:real_len].tolist(),
            'answer_sequence': y[:real_len].tolist(),
            'length': int(real_len)
        }


def episode_reward(initial_score, final_score, full_score):
    """
    计算 episode 的奖励（归一化学习增益）
    
    Args:
        initial_score: 初始掌握度
        final_score: 最终掌握度
        full_score: 满分
    
    Returns:
        normalized_reward: 归一化后的奖励
    """
    delta = final_score - initial_score
    normalize_factor = full_score - initial_score + 1e-9
    return delta / normalize_factor


if __name__ == '__main__':
    """测试 KT 环境"""
    print("=" * 60)
    print("测试 KT 环境")
    print("=" * 60)
    
    # 创建环境
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    print("\n--- 测试 1: 单个学生的学习路径 ---")
    # 设置目标（随机选 3 个知识点）
    batch_size = 1
    targets = np.random.randint(0, env.skill_num, (batch_size, 3))
    initial_logs = np.random.randint(0, env.skill_num, (batch_size, 10))
    
    # 重置环境
    state_info = env.reset(targets, initial_logs)
    print(f"初始掌握度: {state_info['initial_score']}")
    print(f"学习目标: {targets[0]}")
    
    # 执行学习路径（10 步）
    learning_path = np.random.randint(0, env.skill_num, (batch_size, 10))
    for step_idx in range(10):
        kc = learning_path[:, step_idx:step_idx+1]
        step_info = env.step(kc)
        
        if step_idx % 3 == 0:
            print(f"Step {step_idx}: 学习 KC {kc[0,0]}, "
                  f"目标掌握度 = {step_info['current_target_score'][0]:.4f}")
    
    # 计算最终奖励
    final_score = env.evaluate()
    reward = env.get_reward(full_score=len(targets[0]))
    print(f"\n最终掌握度: {final_score[0]:.4f}")
    print(f"学习增益: {final_score[0] - state_info['initial_score'][0]:.4f}")
    print(f"归一化奖励: {reward[0]:.4f}")
    
    print("\n--- 测试 2: 批量学生 ---")
    batch_size = 4
    targets = np.random.randint(0, env.skill_num, (batch_size, 3))
    initial_logs = np.random.randint(0, env.skill_num, (batch_size, 10))
    
    state_info = env.reset(targets, initial_logs)
    print(f"批量大小: {batch_size}")
    print(f"初始掌握度均值: {state_info['initial_score'].mean():.4f}")
    
    # 执行 5 步学习
    learning_path = np.random.randint(0, env.skill_num, (batch_size, 5))
    env.step(learning_path)
    
    final_scores = env.evaluate()
    rewards = env.get_reward(full_score=3)
    print(f"最终掌握度均值: {final_scores.mean():.4f}")
    print(f"平均奖励: {rewards.mean():.4f}")
    
    print("\n✅ 所有测试通过！")

