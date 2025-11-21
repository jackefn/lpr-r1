#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DKT 环境封装 - Concept级别（138个KC）
用于学习路径推荐的强化学习环境

提供标准的 RL 接口：reset(), step(), evaluate()
"""
import os
import json
import numpy as np
from mindspore import Tensor, load_param_into_net, load_checkpoint
from mindspore import dtype as mdtype
from argparse import Namespace

from KTScripts.DataLoader import KTDataset
from KTScripts.utils import load_model


class KTEnvConcepts:
    """
    知识追踪环境 - Concept级别（138个KC）
    将 DKT 模型封装为 RL 环境，支持从真实数据集加载学生状态
    
    功能：
    1. 从数据集加载真实学生的学习记录
    2. 从train/test数据集加载学习目标
    3. 模拟学生学习过程并计算掌握度
    4. 计算学习路径的奖励
    """
    
    def __init__(self, model_path='SavedModels/DKT_assist09_kcs_concept138.ckpt',
                 data_path='data/assist09_kcs/assist09_kcs.npz',
                 train_set_path='data/assist09/train_set_top3.json',
                 test_set_path='data/assist09/test_set_top3.json',
                 skill_num=138):
        """
        初始化 KT 环境
        
        Args:
            model_path: DKT模型路径
            data_path: 数据集路径（npz）
            train_set_path: 训练集路径（JSON）
            test_set_path: 测试集路径（JSON）
            skill_num: KC数量
        """
        print("\n" + "="*70)
        print("初始化 KT 环境 - Concept级别（138个KC）")
        print("="*70)
        
        self.skill_num = skill_num
        
        # 加载数据集
        print(f"\n1. 加载数据集...")
        self.data = np.load(data_path, allow_pickle=True)
        print(f"   ✅ 数据集: {data_path}")
        print(f"   - KC数量: {skill_num}")
        print(f"   - 学生数量: {len(self.data['skill'])}")
        
        # 加载训练/测试集
        print(f"\n2. 加载训练/测试集...")
        self.train_set = self._load_json(train_set_path) if os.path.exists(train_set_path) else []
        self.test_set = self._load_json(test_set_path) if os.path.exists(test_set_path) else []
        print(f"   ✅ 训练集: {len(self.train_set)} 学生")
        print(f"   ✅ 测试集: {len(self.test_set)} 学生")
        
        # 加载训练好的 DKT 模型
        print(f"\n3. 加载DKT模型...")
        self.model = self._load_model(model_path, skill_num)
        
        # 当前状态
        self.current_student_id = None
        self.targets = None
        self.states = (None, None)  # LSTM hidden states
        self.initial_score = None
        self.current_score = None
        self.step_count = 0
        
        print(f"\n{'='*70}")
        print("✅ KT 环境初始化完成！")
        print("="*70 + "\n")
    
    def _load_json(self, path):
        """加载JSON文件"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_model(self, model_path, skill_num):
        """加载训练好的 DKT 模型"""
        print(f"   模型路径: {model_path}")
        
        # 创建模型配置
        args = Namespace(
            model='DKT',
            feat_nums=skill_num,
            embed_size=128,
            hidden_size=128,
            output_size=1,
            dropout=0.3,
            forRec=False,
            pre_hidden_sizes=[256, 64, 16],
            retrieval=False,
            without_label=False
        )
        
        # 创建模型
        model = load_model(args)
        
        # 加载权重
        if os.path.exists(model_path):
            load_param_into_net(model, load_checkpoint(model_path))
            print(f"   ✅ 模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model.set_train(False)  # 设置为评估模式
        return model
    
    def reset(self, student_id=None, targets=None, initial_logs=None, 
              use_train_set=True, random_student=True):
        """
        重置环境，开始新的学习 episode
        
        支持两种模式：
        1. 随机模式：从train/test集中随机选择学生
        2. 指定模式：使用指定的student_id、targets和initial_logs
        
        Args:
            student_id: 学生ID（可选，random_student=False时必需）
            targets: 目标KC列表（可选，使用数据集中的目标）
            initial_logs: 初始学习记录（可选，使用数据集中的记录）
            use_train_set: 是否使用训练集（True）还是测试集（False）
            random_student: 是否随机选择学生
        
        Returns:
            state_info: 包含初始状态信息的字典
        """
        # 选择学生
        if random_student:
            dataset = self.train_set if use_train_set else self.test_set
            if not dataset:
                raise ValueError("数据集为空！请先加载训练/测试集")
            
            student_record = np.random.choice(dataset)
            student_id = student_record['student_id']
            targets = student_record['target_concepts']
            initial_logs = student_record['skills_seq']  # 使用正确的键名
        else:
            if student_id is None:
                raise ValueError("当 random_student=False 时，必须提供 student_id")
            if targets is None or initial_logs is None:
                # 从数据集中获取
                initial_logs = self.data['skill'][student_id].tolist()
                if targets is None:
                    # 随机选择3个未练习的KC
                    practiced_kcs = set(initial_logs)
                    unpracticed_kcs = [kc for kc in range(self.skill_num) 
                                      if kc not in practiced_kcs]
                    targets = np.random.choice(unpracticed_kcs, 
                                              size=min(3, len(unpracticed_kcs)), 
                                              replace=False).tolist()
        
        self.current_student_id = student_id
        self.targets = np.array(targets).reshape(1, -1)  # (1, num_targets)
        self.step_count = 0
        
        # 初始化 LSTM 状态
        self.states = (None, None)
        
        # 使用初始学习记录更新状态
        if initial_logs is not None and len(initial_logs) > 0:
            initial_logs_tensor = Tensor(
                np.array(initial_logs).reshape(1, -1), 
                dtype=mdtype.int32
            )
            _, self.states = self.model.learn_lstm(initial_logs_tensor)
        
        # 计算初始掌握度
        self.initial_score = self._evaluate_targets()
        self.current_score = self.initial_score.copy()
        
        return {
            'student_id': student_id,
            'targets': targets,
            'initial_logs': initial_logs,
            'initial_score': float(self.initial_score),
            'skill_num': self.skill_num,
            'num_targets': len(targets)
        }
    
    def step(self, kc_ids, binary=True):
        """
        执行一步学习：学生学习指定的知识点
        
        Args:
            kc_ids: 要学习的知识点ID列表或单个ID
                   - 单个KC: int
                   - 多个KC: list of int
            binary: 是否将学习结果二值化（>0.5 为掌握）
        
        Returns:
            step_info: 包含学习后状态信息的字典
        """
        # 处理输入格式
        if isinstance(kc_ids, (int, np.integer)):
            kc_ids = [int(kc_ids)]
        elif not isinstance(kc_ids, list):
            kc_ids = list(kc_ids)
        
        kc_array = np.array(kc_ids).reshape(1, -1)  # (1, seq_len)
        kc_tensor = Tensor(kc_array, dtype=mdtype.int32)
        
        # 通过 DKT 模拟学习过程
        scores, self.states = self.model.learn_lstm(kc_tensor, *self.states)
        
        # 是否二值化学习结果
        if binary:
            import mindspore.ops as ops
            scores = ops.cast(scores > 0.5, mdtype.float32)
        
        # 更新步数
        self.step_count += len(kc_ids)
        
        # 计算当前对目标的掌握度
        self.current_score = self._evaluate_targets()
        
        return {
            'kc_ids': kc_ids,
            'learning_scores': scores.asnumpy().flatten().tolist(),
            'current_target_score': float(self.current_score),
            'step_count': self.step_count
        }
    
    def evaluate(self):
        """
        评估当前状态下对目标知识点的掌握度
        
        Returns:
            掌握度分数 (float)
        """
        return float(self.current_score)
    
    def get_reward(self, reward_type='normalized'):
        """
        计算学习路径的奖励
        
        Args:
            reward_type: 奖励类型
                - 'normalized': 归一化学习增益 (E_end - E_start) / (1 - E_start)
                - 'delta': 简单增益 E_end - E_start
                - 'final': 最终掌握度 E_end
        
        Returns:
            reward: float
        """
        if reward_type == 'normalized':
            delta = self.current_score - self.initial_score
            normalize_factor = 1.0 - self.initial_score + 1e-9
            return float(delta / normalize_factor)
        elif reward_type == 'delta':
            return float(self.current_score - self.initial_score)
        elif reward_type == 'final':
            return float(self.current_score)
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
    
    def _evaluate_targets(self):
        """
        评估在当前状态下对目标知识点的掌握度
        
        Returns:
            平均掌握度分数 (numpy scalar)
        """
        import mindspore.ops as ops
        
        targets_tensor = Tensor(self.targets, dtype=mdtype.int32)  # (1, num_targets)
        
        scores = []
        for i in range(targets_tensor.shape[1]):
            # 对每个目标知识点评估掌握度
            target_kc = targets_tensor[:, i:i+1]
            score, _ = self.model.learn_lstm(target_kc, *self.states)
            scores.append(score)
        
        # 返回平均掌握度
        mean_score = ops.mean(ops.concat(scores, axis=1), axis=1)
        return mean_score.asnumpy()[0]
    
    def get_student_practiced_kcs(self, student_id=None):
        """
        获取学生已练习过的KC集合
        
        Args:
            student_id: 学生ID（可选，默认使用当前学生）
        
        Returns:
            set of KC IDs
        """
        if student_id is None:
            student_id = self.current_student_id
        
        if student_id is None:
            return set()
        
        skill_seq = self.data['skill'][student_id]
        return set(skill_seq.tolist())
    
    def get_candidate_kcs(self, exclude_practiced=True, exclude_targets=False):
        """
        获取候选KC列表（用于生成学习路径）
        
        Args:
            exclude_practiced: 是否排除已练习的KC
            exclude_targets: 是否排除目标KC
        
        Returns:
            list of KC IDs
        """
        all_kcs = set(range(self.skill_num))
        
        if exclude_practiced:
            practiced = self.get_student_practiced_kcs()
            all_kcs -= practiced
        
        if exclude_targets and self.targets is not None:
            target_set = set(self.targets.flatten().tolist())
            all_kcs -= target_set
        
        return list(all_kcs)


def test_env():
    """测试 KT 环境"""
    print("\n" + "="*70)
    print("测试 KT 环境 - Concept级别")
    print("="*70)
    
    # 创建环境
    env = KTEnvConcepts()
    
    print("\n--- 测试 1: 从训练集随机选择学生 ---")
    state_info = env.reset(use_train_set=True, random_student=True)
    print(f"学生ID: {state_info['student_id']}")
    print(f"学习目标: {state_info['targets']}")
    print(f"初始学习记录长度: {len(state_info['initial_logs'])}")
    print(f"初始掌握度: {state_info['initial_score']:.4f}")
    
    # 执行学习路径（10 步）
    print(f"\n执行10步学习路径:")
    for step_idx in range(10):
        # 随机选择一个候选KC（排除已练习的）
        candidates = env.get_candidate_kcs(exclude_practiced=False, exclude_targets=False)
        kc = np.random.choice(candidates)
        
        step_info = env.step(kc)
        
        if step_idx % 3 == 0:
            print(f"  Step {step_idx+1}: 学习 KC {kc}, "
                  f"目标掌握度 = {step_info['current_target_score']:.4f}")
    
    # 计算最终奖励
    final_score = env.evaluate()
    reward_normalized = env.get_reward('normalized')
    reward_delta = env.get_reward('delta')
    
    print(f"\n学习结果:")
    print(f"  初始掌握度: {state_info['initial_score']:.4f}")
    print(f"  最终掌握度: {final_score:.4f}")
    print(f"  掌握度提升: {reward_delta:.4f}")
    print(f"  归一化奖励: {reward_normalized:.4f}")
    
    print("\n--- 测试 2: 从测试集选择学生 ---")
    state_info = env.reset(use_train_set=False, random_student=True)
    print(f"学生ID: {state_info['student_id']}")
    print(f"学习目标: {state_info['targets']}")
    print(f"初始掌握度: {state_info['initial_score']:.4f}")
    
    # 执行5步学习，只学习目标KC
    print(f"\n执行5步学习（专注目标KC）:")
    targets = state_info['targets']
    for step_idx in range(5):
        kc = targets[step_idx % len(targets)]
        step_info = env.step(kc)
        print(f"  Step {step_idx+1}: 学习目标KC {kc}, "
              f"掌握度 = {step_info['current_target_score']:.4f}")
    
    final_score = env.evaluate()
    reward = env.get_reward('normalized')
    print(f"\n学习结果:")
    print(f"  初始掌握度: {state_info['initial_score']:.4f}")
    print(f"  最终掌握度: {final_score:.4f}")
    print(f"  归一化奖励: {reward:.4f}")
    
    print("\n--- 测试 3: 批量学习（一次学习多个KC） ---")
    state_info = env.reset(use_train_set=True, random_student=True)
    print(f"学生ID: {state_info['student_id']}")
    print(f"初始掌握度: {state_info['initial_score']:.4f}")
    
    # 一次学习3个KC
    candidates = env.get_candidate_kcs()
    kc_batch = np.random.choice(candidates, size=3, replace=False).tolist()
    step_info = env.step(kc_batch)
    
    print(f"\n一次学习3个KC: {kc_batch}")
    print(f"学习后掌握度: {step_info['current_target_score']:.4f}")
    print(f"总步数: {step_info['step_count']}")
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！")
    print("="*70 + "\n")


if __name__ == '__main__':
    """运行测试"""
    test_env()

