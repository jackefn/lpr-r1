#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 DKT 环境的完整功能
"""
import numpy as np
from kt_env import KTEnv


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试 1: 基本功能")
    print("=" * 60)
    
    # 创建环境
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    # 设置一个学习场景
    batch_size = 2
    num_targets = 3
    initial_len = 10
    path_length = 20
    
    # 随机生成目标和初始记录
    targets = np.random.randint(0, env.skill_num, (batch_size, num_targets))
    initial_logs = np.random.randint(0, env.skill_num, (batch_size, initial_len))
    
    print(f"\n配置:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - 目标数量: {num_targets}")
    print(f"  - 初始记录长度: {initial_len}")
    print(f"  - 学习路径长度: {path_length}")
    
    # 重置环境
    state_info = env.reset(targets, initial_logs)
    print(f"\n初始状态:")
    for i in range(batch_size):
        print(f"  学生 {i}: 目标 {targets[i]}, 初始掌握度 {state_info['initial_score'][i]:.4f}")
    
    # 执行学习路径
    print(f"\n执行学习路径...")
    learning_path = np.random.randint(0, env.skill_num, (batch_size, path_length))
    
    for step in range(path_length):
        kc = learning_path[:, step:step+1]
        step_info = env.step(kc)
        
        if step % 5 == 0:
            print(f"  Step {step:2d}: 目标掌握度均值 = {step_info['current_target_score'].mean():.4f}")
    
    # 计算最终结果
    final_scores = env.evaluate()
    rewards = env.get_reward(full_score=num_targets)
    
    print(f"\n最终结果:")
    for i in range(batch_size):
        print(f"  学生 {i}: 最终掌握度 {final_scores[i]:.4f}, "
              f"增益 {final_scores[i] - state_info['initial_score'][i]:.4f}, "
              f"奖励 {rewards[i]:.4f}")
    
    print(f"\n✅ 测试 1 通过")
    return env


def test_incremental_learning():
    """测试增量学习"""
    print("\n" + "=" * 60)
    print("测试 2: 增量学习（逐步添加知识点）")
    print("=" * 60)
    
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    batch_size = 1
    targets = np.random.randint(0, env.skill_num, (batch_size, 3))
    
    # 不提供初始记录
    state_info = env.reset(targets, initial_logs=None)
    print(f"\n无初始记录，初始掌握度: {state_info['initial_score'][0]:.4f}")
    
    # 逐个学习知识点
    print(f"\n逐步学习 10 个知识点:")
    for i in range(10):
        kc = np.random.randint(0, env.skill_num, (batch_size, 1))
        step_info = env.step(kc)
        print(f"  学习 KC {kc[0,0]:5d} → 目标掌握度 {step_info['current_target_score'][0]:.4f}")
    
    final_score = env.evaluate()[0]
    reward = env.get_reward(full_score=3)[0]
    print(f"\n最终: 掌握度 {final_score:.4f}, 奖励 {reward:.4f}")
    print(f"✅ 测试 2 通过")


def test_different_targets():
    """测试不同的目标设置"""
    print("\n" + "=" * 60)
    print("测试 3: 不同数量的目标")
    print("=" * 60)
    
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    batch_size = 1
    initial_logs = np.random.randint(0, env.skill_num, (batch_size, 10))
    
    for num_targets in [1, 3, 5]:
        print(f"\n--- {num_targets} 个目标 ---")
        targets = np.random.randint(0, env.skill_num, (batch_size, num_targets))
        
        state_info = env.reset(targets, initial_logs)
        print(f"初始掌握度: {state_info['initial_score'][0]:.4f}")
        
        # 学习 15 步
        learning_path = np.random.randint(0, env.skill_num, (batch_size, 15))
        env.step(learning_path)
        
        final_score = env.evaluate()[0]
        reward = env.get_reward(full_score=num_targets)[0]
        print(f"最终掌握度: {final_score:.4f}")
        print(f"奖励: {reward:.4f}")
    
    print(f"\n✅ 测试 3 通过")


def test_real_student_data():
    """测试使用真实学生数据"""
    print("\n" + "=" * 60)
    print("测试 4: 使用真实学生数据")
    print("=" * 60)
    
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    # 获取前 3 个学生的真实数据
    print(f"\n数据集包含 {len(env.dataset)} 个学生")
    
    for student_id in range(min(3, len(env.dataset))):
        student_data = env.get_student_data(student_id)
        print(f"\n学生 {student_id}:")
        print(f"  - 记录长度: {student_data['length']}")
        print(f"  - 前 10 个技能: {student_data['skill_sequence'][:10]}")
        print(f"  - 前 10 个答题: {student_data['answer_sequence'][:10]}")
    
    print(f"\n✅ 测试 4 通过")


def test_batch_processing():
    """测试批量处理效率"""
    print("\n" + "=" * 60)
    print("测试 5: 批量处理")
    print("=" * 60)
    
    env = KTEnv(model_name='DKT', dataset_name='assist09')
    
    import time
    
    for batch_size in [1, 8, 32]:
        print(f"\n--- Batch size: {batch_size} ---")
        
        targets = np.random.randint(0, env.skill_num, (batch_size, 3))
        initial_logs = np.random.randint(0, env.skill_num, (batch_size, 10))
        learning_path = np.random.randint(0, env.skill_num, (batch_size, 20))
        
        start_time = time.time()
        
        env.reset(targets, initial_logs)
        env.step(learning_path)
        rewards = env.get_reward(full_score=3)
        
        elapsed = time.time() - start_time
        
        print(f"  耗时: {elapsed:.4f} 秒")
        print(f"  平均奖励: {rewards.mean():.4f}")
        print(f"  每个样本耗时: {elapsed/batch_size*1000:.2f} ms")
    
    print(f"\n✅ 测试 5 通过")


if __name__ == '__main__':
    print("\n" + "🚀 " * 20)
    print(" " * 20 + "DKT 环境完整测试")
    print("🚀 " * 20 + "\n")
    
    try:
        # 运行所有测试
        env = test_basic_functionality()
        test_incremental_learning()
        test_different_targets()
        test_real_student_data()
        test_batch_processing()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！DKT 环境工作正常！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

