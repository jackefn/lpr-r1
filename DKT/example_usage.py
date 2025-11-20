#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DKT 环境使用示例
演示如何使用 DKT 环境进行学习路径推荐
"""
import numpy as np
from kt_env import KTEnv


def example_1_basic_usage():
    """示例 1: 基本使用流程"""
    print("=" * 60)
    print("示例 1: 基本使用流程")
    print("=" * 60)
    
    # 1. 创建环境
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    # 2. 设置学习场景
    batch_size = 1
    targets = np.array([[100, 200, 300]])  # 3 个目标知识点
    initial_logs = np.array([[50, 51, 52, 53, 54]])  # 5 条历史记录
    
    print(f"\n学习目标: {targets[0]}")
    print(f"初始历史: {initial_logs[0]}")
    
    # 3. 重置环境
    state = env.reset(targets, initial_logs)
    print(f"\n初始掌握度: {state['initial_score'][0]:.4f}")
    
    # 4. 执行学习路径
    learning_path = np.array([[101, 102, 103, 201, 202, 203, 301, 302, 303, 104]]).T
    print(f"\n学习路径: {learning_path.T[0]}")
    
    for i, kc in enumerate(learning_path):
        step_info = env.step(kc.reshape(1, 1))
        print(f"  Step {i+1}: 学习 KC {kc[0]:3d} → 掌握度 {step_info['current_target_score'][0]:.4f}")
    
    # 5. 计算最终结果
    final_score = env.evaluate()[0]
    reward = env.get_reward(full_score=3)[0]
    
    print(f"\n最终掌握度: {final_score:.4f}")
    print(f"学习增益: {final_score - state['initial_score'][0]:+.4f}")
    print(f"归一化奖励: {reward:.4f}")


def example_2_compare_paths():
    """示例 2: 比较不同学习路径的效果"""
    print("\n" + "=" * 60)
    print("示例 2: 比较不同学习路径")
    print("=" * 60)
    
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    # 设置相同的初始条件
    targets = np.array([[1000, 2000, 3000]])
    initial_logs = np.array([[100, 101, 102, 103, 104]])
    
    # 路径 A: 循序渐进（接近目标）
    path_a = np.array([[950, 980, 990, 995, 1000,
                        1950, 1980, 1990, 1995, 2000,
                        2950, 2980, 2990, 2995, 3000]]).T
    
    # 路径 B: 随机选择
    np.random.seed(42)
    path_b = np.random.randint(0, env.skill_num, (15, 1))
    
    # 评估路径 A
    state = env.reset(targets, initial_logs)
    env.step(path_a)
    reward_a = env.get_reward(full_score=3)[0]
    
    # 评估路径 B
    env.reset(targets, initial_logs)
    env.step(path_b)
    reward_b = env.get_reward(full_score=3)[0]
    
    print(f"\n路径 A（循序渐进）:")
    print(f"  奖励: {reward_a:.4f}")
    
    print(f"\n路径 B（随机选择）:")
    print(f"  奖励: {reward_b:.4f}")
    
    print(f"\n路径 A 比路径 B {('更好' if reward_a > reward_b else '更差')}")
    print(f"差异: {abs(reward_a - reward_b):.4f}")


def example_3_batch_students():
    """示例 3: 批量处理多个学生"""
    print("\n" + "=" * 60)
    print("示例 3: 批量处理多个学生")
    print("=" * 60)
    
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    # 为 5 个学生生成不同的学习场景
    batch_size = 5
    np.random.seed(123)
    
    # 每个学生有不同的目标
    targets = np.random.randint(0, env.skill_num, (batch_size, 3))
    
    # 每个学生有不同的历史
    initial_logs = np.random.randint(0, env.skill_num, (batch_size, 8))
    
    print(f"\n{batch_size} 个学生的学习场景:")
    state = env.reset(targets, initial_logs)
    
    for i in range(batch_size):
        print(f"  学生 {i+1}: 目标 {targets[i]}, 初始掌握度 {state['initial_score'][i]:.4f}")
    
    # 执行相同的学习路径（简化）
    learning_path = np.random.randint(0, env.skill_num, (batch_size, 12))
    env.step(learning_path)
    
    # 计算结果
    rewards = env.get_reward(full_score=3)
    
    print(f"\n学习结果:")
    for i in range(batch_size):
        print(f"  学生 {i+1}: 奖励 {rewards[i]:.4f}")
    
    print(f"\n平均奖励: {rewards.mean():.4f}")


def example_4_real_student():
    """示例 4: 使用真实学生数据"""
    print("\n" + "=" * 60)
    print("示例 4: 基于真实学生数据")
    print("=" * 60)
    
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    # 获取真实学生的数据
    student_id = 0
    student_data = env.get_student_data(student_id)
    
    print(f"\n真实学生 {student_id}:")
    print(f"  历史长度: {student_data['length']}")
    print(f"  技能序列前 10: {student_data['skill_sequence'][:10]}")
    
    # 使用学生的前 10 条记录作为初始历史
    initial_logs = np.array([student_data['skill_sequence'][:10]])
    
    # 从学生的技能序列中选择 3 个作为目标
    targets = np.array([student_data['skill_sequence'][10:13]])
    
    print(f"\n学习目标: {targets[0]}")
    
    # 重置环境
    state = env.reset(targets, initial_logs)
    print(f"初始掌握度: {state['initial_score'][0]:.4f}")
    
    # 使用学生实际学习的下 10 个技能
    actual_path = np.array([student_data['skill_sequence'][13:23]]).T
    env.step(actual_path)
    
    reward = env.get_reward(full_score=3)[0]
    print(f"学生实际路径的奖励: {reward:.4f}")


def example_5_adaptive_learning():
    """示例 5: 自适应学习路径（根据掌握度动态调整）"""
    print("\n" + "=" * 60)
    print("示例 5: 自适应学习路径")
    print("=" * 60)
    
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    # 设置学习目标
    targets = np.array([[500, 1000, 1500]])
    initial_logs = np.array([[10, 20, 30, 40, 50]])
    
    state = env.reset(targets, initial_logs)
    print(f"\n初始掌握度: {state['initial_score'][0]:.4f}")
    print(f"学习目标: {targets[0]}")
    
    print(f"\n自适应学习过程:")
    
    # 模拟一个简单的自适应策略：
    # 如果掌握度低，学习相关基础知识；如果掌握度高，学习进阶知识
    for step in range(10):
        current_score = env.evaluate()[0]
        
        # 简单策略：根据当前掌握度选择知识点
        if current_score < 0.3:
            # 掌握度低，学习基础知识（接近目标但更简单）
            kc = targets[0, step % 3] - 50
        elif current_score < 0.6:
            # 掌握度中等，学习核心知识（目标本身）
            kc = targets[0, step % 3]
        else:
            # 掌握度高，学习进阶知识（超过目标）
            kc = targets[0, step % 3] + 50
        
        kc = max(0, min(kc, env.skill_num - 1))  # 确保在有效范围内
        
        step_info = env.step(np.array([[kc]]))
        new_score = step_info['current_target_score'][0]
        
        print(f"  Step {step+1}: 掌握度 {current_score:.4f} → 学习 KC {kc:4d} → 新掌握度 {new_score:.4f}")
    
    reward = env.get_reward(full_score=3)[0]
    print(f"\n最终奖励: {reward:.4f}")


if __name__ == '__main__':
    print("\n" + "📚 " * 20)
    print(" " * 20 + "DKT 环境使用示例")
    print("📚 " * 20 + "\n")
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_compare_paths()
    example_3_batch_students()
    example_4_real_student()
    example_5_adaptive_learning()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)

