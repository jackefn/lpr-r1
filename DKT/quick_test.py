#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证 DKT 环境是否正常工作
"""
import sys
import numpy as np

def quick_test():
    """快速测试 DKT 环境的核心功能"""
    print("🔍 快速测试 DKT 环境...")
    print("-" * 60)
    
    try:
        # 1. 导入模块
        print("1. 导入 kt_env 模块...", end=" ")
        from kt_env import KTEnv
        print("✅")
        
        # 2. 创建环境
        print("2. 创建 DKT 环境...", end=" ")
        env = KTEnv(model_name='DKT', dataset_name='assist09')
        print("✅")
        
        # 3. 测试 reset
        print("3. 测试 reset()...", end=" ")
        targets = np.random.randint(0, env.skill_num, (2, 3))
        initial_logs = np.random.randint(0, env.skill_num, (2, 10))
        state = env.reset(targets, initial_logs)
        assert 'initial_score' in state
        assert len(state['initial_score']) == 2
        print("✅")
        
        # 4. 测试 step
        print("4. 测试 step()...", end=" ")
        kc = np.random.randint(0, env.skill_num, (2, 1))
        step_info = env.step(kc)
        assert 'current_target_score' in step_info
        assert len(step_info['current_target_score']) == 2
        print("✅")
        
        # 5. 测试 evaluate
        print("5. 测试 evaluate()...", end=" ")
        scores = env.evaluate()
        assert len(scores) == 2
        print("✅")
        
        # 6. 测试 get_reward
        print("6. 测试 get_reward()...", end=" ")
        rewards = env.get_reward(full_score=3)
        assert len(rewards) == 2
        print("✅")
        
        # 7. 测试数据加载
        print("7. 测试 get_student_data()...", end=" ")
        student_data = env.get_student_data(0)
        assert 'skill_sequence' in student_data
        assert 'length' in student_data
        print("✅")
        
        print("-" * 60)
        print("🎉 所有核心功能测试通过！")
        print("\n环境信息:")
        print(f"  - 技能数量: {env.skill_num}")
        print(f"  - 数据集大小: {len(env.dataset)} 个学生")
        print(f"  - 初始掌握度范围: [{state['initial_score'].min():.3f}, {state['initial_score'].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)

