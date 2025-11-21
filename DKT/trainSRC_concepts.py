#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SRC训练脚本 - Concept级别（138个KC）
基于原trainSRC.py，适配concept级别的DKT模型
"""
# Copyright 2023 Huawei Technologies Co., Ltd
import os
import sys
import time
from argparse import ArgumentParser, Namespace

import numpy as np
from mindspore import load_param_into_net, load_checkpoint, save_checkpoint, context
from mindspore.nn import PolynomialDecayLR, Adam
from tqdm import tqdm

# 添加父目录到路径以导入KTScripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from KTScripts.DataLoader import KTDataset
from KTScripts.utils import set_random_seed


def get_concept_level_options(parser: ArgumentParser):
    """
    获取concept级别的配置选项
    """
    # Agent和模型选择
    agent_choices = ['MPC', 'DQN', 'SRC']
    model_choices = ['DKT', 'CoKT']
    parser.add_argument('-a', '--agent', type=str, choices=agent_choices, default='SRC',
                       help='Agent类型')
    parser.add_argument('-m', '--model', type=str, choices=model_choices, default='DKT',
                       help='知识追踪模型')
    parser.add_argument('-d', '--dataset', type=str, default='assist09_kcs',
                       help='Concept级别数据集（138个KC）')
    
    # 训练配置
    parser.add_argument('-w', '--worker', type=int, default=6)
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                       help='批次大小（concept级别建议128）')
    parser.add_argument('-p', '--path', type=int, default=0, choices=[0, 1, 2, 3],
                       help='学习路径类型')
    parser.add_argument('-s', '--steps', type=int, default=10,
                       help='每个episode的学习步数')
    parser.add_argument('--num_targets', type=int, default=3,
                       help='目标KC数量')
    parser.add_argument('--num_initial_logs', type=int, default=10,
                       help='初始学习记录数量')
    
    # 路径配置
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./SavedModels')
    parser.add_argument('--visual_dir', type=str, default='./VisualResults')
    parser.add_argument('--dkt_model_path', type=str, 
                       default='./SavedModels/DKT_assist09_kcs_concept138.ckpt',
                       help='Concept级别DKT模型路径')
    
    # 优化器配置
    parser.add_argument('-c', '--cuda', type=int, default=-1,
                       help='GPU设备ID，-1表示使用CPU')
    parser.add_argument('-e', '--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--decay_steps', type=int, default=200)
    parser.add_argument('--l2_reg', type=float, default=1e-4)
    
    # 模型配置
    parser.add_argument('--embed_size', type=int, default=128,
                       help='Embedding维度（对应concept级别）')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--predict_hidden_sizes', type=int, nargs='+',
                       default=[256, 64, 16])
    
    # 其他配置
    parser.add_argument('--load_model', action='store_true',
                       help='是否加载已有的Agent模型')
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--train_iterations', type=int, default=200,
                       help='每个epoch的训练迭代次数')
    parser.add_argument('--test_iterations', type=int, default=200,
                       help='测试迭代次数')
    
    args = parser.parse_args().__dict__
    args = Namespace(**args)
    
    # 生成实验名称
    args.exp_name = f'{args.agent}_{args.dataset}_path{args.path}_concept138'
    
    # 设置运行环境
    if args.cuda >= 0:
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=args.cuda)
        print(f"使用GPU: {args.cuda}")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        print("使用CPU")
    
    return args


class ConceptKESEnv:
    """
    Concept级别的知识环境（Knowledge Evolution Simulator）
    基于concept级别（138个KC）的DKT模型
    """
    def __init__(self, dataset, dkt_model_path, skill_num=138):
        from mindspore import dtype as mdtype, jit as ms_function
        from mindspore import ops
        
        self.skill_num = skill_num  # 138个KC
        self.ops = ops
        self.mdtype = mdtype
        
        # 加载concept级别的DKT模型
        print(f"\n加载Concept级别DKT模型:")
        print(f"  模型路径: {dkt_model_path}")
        print(f"  KC数量: {self.skill_num}")
        
        self.model = self._load_dkt_model(dkt_model_path, skill_num)
        self.targets = None
        self.states = (None, None)
        self.initial_score = None
    
    def _load_dkt_model(self, model_path, skill_num):
        """加载DKT模型"""
        from KTScripts.utils import load_model
        
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
        
        model = load_model(args)
        
        if os.path.exists(model_path):
            load_param_into_net(model, load_checkpoint(model_path))
            print(f"  ✅ 模型加载成功\n")
        else:
            print(f"  ⚠️  模型文件不存在: {model_path}")
            print(f"     使用未训练的模型\n")
        
        return model
    
    def exam(self, targets, states):
        """
        评估目标KCs的掌握度
        Args:
            targets: [batch_size, num_targets] - 目标KC IDs
            states: (h, c) - LSTM隐藏状态
        Returns:
            scores: [batch_size] - 平均掌握度
        """
        scores = []
        for i in range(targets.shape[1]):
            score, _ = self.model.learn_lstm(targets[:, i:i + 1], *states)
            scores.append(score)
        return self.ops.mean(self.ops.concat(scores, axis=1), axis=1)
    
    def begin_episode(self, targets, initial_logs):
        """
        开始一个学习episode
        Args:
            targets: [batch_size, num_targets] - 目标KC IDs
            initial_logs: [batch_size, log_len] - 初始学习记录
        Returns:
            initial_log_scores: 初始学习记录的掌握度
        """
        self.targets = targets
        initial_score, initial_log_scores, states = self._begin_episode(targets, initial_logs)
        self.initial_score = initial_score
        self.states = states
        return initial_log_scores
    
    def _begin_episode(self, targets, initial_logs=None):
        """内部实现"""
        states = (None, None)
        score = None
        if initial_logs is not None:
            score, states = self.model.learn_lstm(initial_logs)
        initial_score = self.exam(targets, states)
        return initial_score, score, states
    
    def n_step(self, learning_path, binary=False):
        """
        执行n步学习
        Args:
            learning_path: [batch_size, steps] - 学习路径（KC IDs）
            binary: 是否二值化答题结果
        Returns:
            scores: 每步的掌握度
        """
        scores, states = self.model.learn_lstm(learning_path, *self.states)
        self.states = states
        if binary:
            scores = self.ops.cast(scores > 0.5, self.mdtype.float32)
        return scores
    
    def end_episode(self, return_score=False):
        """
        结束episode，计算reward
        Args:
            return_score: 是否返回最终分数
        Returns:
            reward: [batch_size, 1] - 学习效果（掌握度提升）
            或 (final_score, reward)
        """
        final_score = self.exam(self.targets, self.states)
        # reward = (final_score - initial_score) / (1.0 - initial_score)
        reward = self._episode_reward(self.initial_score, final_score, 1.0)
        reward = reward.expand_dims(-1)
        
        if return_score:
            return final_score, reward
        return reward
    
    def _episode_reward(self, initial_score, final_score, full_score):
        """计算episode奖励（掌握度提升）"""
        return (final_score - initial_score) / (full_score - initial_score + 1e-8)


def get_concept_data(batch_size, skill_num, num_targets=3, num_initial_logs=10, path_type=0, steps=10):
    """
    生成concept级别的训练数据
    Args:
        batch_size: 批次大小
        skill_num: KC总数（138）
        num_targets: 目标KC数量
        num_initial_logs: 初始学习记录数量
        path_type: 路径生成策略
        steps: 学习步数
    Returns:
        targets: [batch_size, num_targets]
        initial_logs: [batch_size, num_initial_logs]
        origin_path: [batch_size, steps] - 参考路径（可选）
    """
    import mindspore as ms
    
    # 为每个样本随机选择目标KCs
    targets = np.random.randint(0, skill_num, size=(batch_size, num_targets))
    
    # 生成初始学习记录（避免与目标KC重复）
    initial_logs = []
    for i in range(batch_size):
        target_set = set(targets[i])
        available_kcs = [kc for kc in range(skill_num) if kc not in target_set]
        if len(available_kcs) < num_initial_logs:
            # 如果可用KC不足，允许重复
            logs = np.random.choice(available_kcs, num_initial_logs, replace=True)
        else:
            logs = np.random.choice(available_kcs, num_initial_logs, replace=False)
        initial_logs.append(logs)
    initial_logs = np.array(initial_logs)
    
    # 生成参考路径（根据path_type）
    if path_type == 0:
        # 随机路径
        origin_path = np.random.randint(0, skill_num, size=(batch_size, steps))
    elif path_type == 1:
        # 重复目标KC的路径
        origin_path = np.tile(targets[:, :1], (1, steps))
    elif path_type == 2:
        # 目标KC的循环路径
        origin_path = np.tile(targets, (1, (steps // num_targets) + 1))[:, :steps]
    else:
        # 混合路径
        origin_path = np.random.randint(0, skill_num, size=(batch_size, steps))
        for i in range(batch_size):
            # 50%的步骤选择目标KC
            target_indices = np.random.choice(steps, steps // 2, replace=False)
            origin_path[i, target_indices] = np.random.choice(targets[i], steps // 2)
    
    # 转换为MindSpore Tensor
    targets = ms.Tensor(targets, dtype=ms.int32)
    initial_logs = ms.Tensor(initial_logs, dtype=ms.int32)
    origin_path = ms.Tensor(origin_path, dtype=ms.int32)
    
    return targets, initial_logs, origin_path


def main(args: Namespace):
    print("\n" + "="*70)
    print(f"SRC训练 - Concept级别（138个KC）")
    print("="*70)
    print(f"Agent: {args.agent}")
    print(f"数据集: {args.dataset}")
    print(f"DKT模型: {args.dkt_model_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习路径类型: {args.path}")
    print(f"学习步数: {args.steps}")
    print(f"目标KC数量: {args.num_targets}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {args.lr}")
    print("="*70 + "\n")
    
    set_random_seed(args.rand_seed)
    
    # 加载数据集（用于获取skill_num等信息）
    dataset = KTDataset(os.path.join(args.data_dir, args.dataset))
    skill_num = dataset.feats_num  # 应该是138
    
    print(f"✅ 数据集加载成功:")
    print(f"   - KC数量: {skill_num}")
    print(f"   - 学生数量: {dataset.users_num}\n")
    
    if skill_num != 138:
        print(f"⚠️  警告: 期望138个KC，实际获得{skill_num}个\n")
    
    # 创建Concept级别的环境
    env = ConceptKESEnv(dataset, args.dkt_model_path, skill_num)
    args.skill_num = skill_num
    
    # 创建Agent（SRC/DQN/MPC）
    print(f"创建Agent: {args.agent}")
    model = load_agent(args)
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    model_path = os.path.join(args.save_dir, f'{args.exp_name}')
    
    # 加载已有模型（如果指定）
    if args.load_model and os.path.exists(f'{model_path}.ckpt'):
        load_param_into_net(model, load_checkpoint(f'{model_path}.ckpt'))
        print(f"✅ 加载Agent模型: {model_path}.ckpt\n")
    else:
        print(f"从头开始训练Agent\n")
    
    # 优化器
    polynomial_decay_lr = PolynomialDecayLR(
        learning_rate=args.lr,
        end_learning_rate=args.min_lr,
        decay_steps=args.decay_steps,
        power=0.5,
        update_decay_steps=True
    )
    optimizer = Adam(model.trainable_params(), learning_rate=polynomial_decay_lr)
    
    # 损失函数和训练封装
    from Scripts.Agent.utils import pl_loss
    from Scripts.Optimizer import ModelWithLoss, ModelWithOptimizer
    
    criterion = pl_loss
    model_with_loss = ModelWithLoss(model, criterion)
    model_train = ModelWithOptimizer(model_with_loss, optimizer)
    
    # 训练
    all_mean_rewards = []
    all_rewards = []
    best_reward = -1e9
    
    print('-' * 70)
    print("开始训练")
    print('-' * 70)
    
    model_train.set_train()
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print('='*70)
        
        avg_time = 0
        epoch_mean_rewards = []
        
        for i in tqdm(range(args.train_iterations), desc=f"训练 Epoch {epoch+1}"):
            t0 = time.perf_counter()
            
            # 生成训练数据
            targets, initial_logs, origin_path = get_concept_data(
                args.batch_size, args.skill_num, 
                args.num_targets, args.num_initial_logs, 
                args.path, args.steps
            )
            
            # 开始episode
            initial_log_scores = env.begin_episode(targets, initial_logs)
            
            # Agent生成学习路径
            data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
            result = model(*data)  # result: (learning_path, value, log_probs)
            
            # 在环境中执行学习路径
            env.n_step(result[0], binary=True)
            
            # 获取reward
            rewards = env.end_episode()
            
            # 更新Agent
            loss = model_train(*data[:-1], result[2], rewards).asnumpy()
            
            mean_reward = np.mean(rewards.asnumpy())
            avg_time += time.perf_counter() - t0
            
            epoch_mean_rewards.append(mean_reward)
            all_rewards.append(mean_reward)
            
            # 每50个batch打印一次
            if (i + 1) % 50 == 0 or i == 0:
                print(f'  batch {i+1}/{args.train_iterations} | '
                      f'avg_time: {avg_time/(i+1):.4f}s | '
                      f'loss: {loss:.4f} | '
                      f'reward: {mean_reward:.4f}')
        
        # Epoch统计
        epoch_mean_reward = np.mean(epoch_mean_rewards)
        all_mean_rewards.append(epoch_mean_reward)
        
        print(f"\nEpoch {epoch+1} 统计:")
        print(f"  平均reward: {epoch_mean_reward:.4f}")
        
        # 保存最佳模型
        if epoch_mean_reward > best_reward:
            best_reward = epoch_mean_reward
            save_checkpoint(model, f"{model_path}.ckpt")
            print(f"  ✅ 新的最佳模型已保存! (Reward: {best_reward:.4f})")
        
        print(f"  当前最佳reward: {best_reward:.4f}")
    
    # 保存训练曲线
    if not os.path.exists(args.visual_dir):
        os.makedirs(args.visual_dir)
    np.save(os.path.join(args.visual_dir, f'{args.exp_name}_rewards'), 
            np.array(all_rewards))
    
    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"最佳reward: {best_reward:.4f}")
    print(f"模型保存: {model_path}.ckpt")
    print('='*70)
    
    # 测试
    print(f"\n{'-'*70}")
    print("开始测试")
    print('-'*70)
    
    test_rewards = []
    model_with_loss.set_train(False)
    load_param_into_net(model, load_checkpoint(f'{model_path}.ckpt'))
    
    for i in tqdm(range(args.test_iterations), desc="测试"):
        targets, initial_logs, origin_path = get_concept_data(
            args.batch_size, args.skill_num,
            args.num_targets, args.num_initial_logs,
            args.path, args.steps
        )
        
        initial_log_scores = env.begin_episode(targets, initial_logs)
        data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
        result = model(*data)
        env.n_step(result[0], binary=True)
        rewards = env.end_episode()
        
        mean_reward = np.mean(rewards.asnumpy())
        test_rewards.append(mean_reward)
        
        if (i + 1) % 50 == 0:
            print(f'  batch {i+1}/{args.test_iterations} | reward: {mean_reward:.4f}')
    
    mean_test_reward = np.mean(test_rewards)
    print(f"\n测试结果:")
    print(f"  平均reward: {mean_test_reward:.4f}")
    print('-'*70 + "\n")


def load_agent(args):
    """加载Agent模型"""
    print(f"  加载Agent: {args.agent}")
    print(f"  Skill数量: {args.skill_num}")
    print(f"  Embedding维度: {args.embed_size}")
    print(f"  隐藏层维度: {args.hidden_size}\n")
    
    # 导入Agent
    if args.agent == 'SRC':
        from Scripts.Agent.SRC import SRC
        # SRC参数: skill_num, input_size, weight_size, hidden_size, dropout, allow_repeat, with_kt
        return SRC(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            weight_size=args.embed_size // 2,  # weight_size通常是input_size的一半
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            allow_repeat=False,
            with_kt=False
        )
    elif args.agent == 'DQN':
        # DQN需要不同的参数
        raise NotImplementedError("DQN agent未实现，请使用SRC")
    elif args.agent == 'MPC':
        # MPC需要不同的参数
        raise NotImplementedError("MPC agent未实现，请使用SRC")
    else:
        raise ValueError(f"Unknown agent: {args.agent}")


if __name__ == '__main__':
    parser = ArgumentParser("SRC Training - Concept Level (138 KCs)")
    args_ = get_concept_level_options(parser)
    main(args_)

