"""
DKT Tool 单元测试
测试LearningPathEnv和DKTUpdateTool的功能
不需要LLM，使用模拟数据
"""
import sys
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from agent.tool.learning_path_env import LearningPathEnv
from agent.tool.tools.dkt_update_tool import DKTUpdateTool


def create_mock_student_data(num_students=3):
    """
    创建模拟学生数据
    
    Returns:
        学生数据列表
    """
    np.random.seed(42)
    
    students = []
    for i in range(num_students):
        # 模拟历史练习序列
        seq_len = np.random.randint(50, 150)
        skills_seq = np.random.randint(0, 138, size=seq_len).tolist()
        corrects_seq = (np.random.random(seq_len) > 0.3).astype(int).tolist()
        
        # 目标概念
        target_kcs = [
            {'id': 122, 'name': 'Sum of Interior Angles Triangle', 'mastery': 0.3},
            {'id': 76, 'name': 'Ordering Integers', 'mastery': 0.25},
            {'id': 105, 'name': 'Reflection', 'mastery': 0.35}
        ]
        
        students.append({
            'student_id': f'student_{i}',
            'skills_seq': skills_seq,
            'corrects_seq': corrects_seq,
            'target_concepts': target_kcs,
            'sequence_length': seq_len
        })
    
    return students


def test_environment_initialization():
    """测试1: 环境初始化"""
    print("\n" + "="*80)
    print("测试1: 环境初始化")
    print("="*80)
    
    # 尝试加载真实的DKT模型
    dkt_model_path = "/mnt/hpfs/xiangc/mxy/lpr-r1/DKT/SavedModels/SRC_assist09_kcs_path0_concept138.ckpt"
    
    # 创建环境
    env = LearningPathEnv(
        dkt_model_path=dkt_model_path,
        num_concepts=138  # ASSIST09数据集有138个概念
    )
    
    print("✅ 环境创建成功")
    
    # 创建模拟学生数据
    students = create_mock_student_data(3)
    
    # 初始化batch
    env.reset_batch(students)
    
    print(f"✅ 初始化了 {len(students)} 个学生")
    
    # 检查状态
    for student in students:
        student_id = student['student_id']
        state = env.get_state(student_id)
        
        assert state is not None, f"Student {student_id} state is None"
        assert 'skills_seq' in state
        assert 'corrects_seq' in state
        assert 'target_concepts' in state
        assert 'initial_mastery' in state
        
        print(f"  - {student_id}: {len(state['skills_seq'])} 历史步骤, "
              f"{len(state['target_concepts'])} 目标概念")
        print(f"    初始掌握度: {state['initial_mastery']}")
    
    return env, students


def test_single_step():
    """测试2: 单步学习"""
    print("\n" + "="*80)
    print("测试2: 单步学习")
    print("="*80)
    
    # 创建环境和学生
    env, students = test_environment_initialization()
    
    # 模拟推荐一个KC（使用真实的ASSIST09概念）
    student_id = students[0]['student_id']
    recommended_kc = "Ordering Integers"  # 真实概念
    
    # 准备概念映射（使用真实的ASSIST09概念）
    concept_map = {
        "Ordering Integers": 76,
        "Sum of Interior Angles Triangle": 122,
        "Reflection": 105,
        "Area Rectangle": 5,
        "Absolute Value": 2,
        "Angle Measurement": 8
    }
    
    print(f"\n推荐: {student_id} 学习 '{recommended_kc}'")
    
    # 执行step
    result = env.step(student_id, recommended_kc, concept_map)
    
    assert result['success'], f"Step failed: {result.get('error')}"
    
    print(f"✅ Step成功")
    print(f"  - 学习结果: {'掌握' if result['correctness'] == 1 else '需要更多练习'}")
    print(f"  - 新的掌握度: {result['new_mastery']}")
    print(f"\n观察结果:")
    print(result['observation'])
    
    # 检查状态更新
    state = env.get_state(student_id)
    assert len(state['learned_in_episode']) == 1
    assert state['learned_in_episode'][0]['kc_name'] == recommended_kc
    
    print(f"\n✅ 状态更新正确")
    
    return env, students, concept_map


def test_multiple_steps():
    """测试3: 多步学习"""
    print("\n" + "="*80)
    print("测试3: 多步学习（完整学习路径）")
    print("="*80)
    
    # 创建环境和学生
    env, students, concept_map = test_single_step()
    
    student_id = students[0]['student_id']
    
    # 模拟一个学习路径（使用真实的ASSIST09概念）
    learning_path = ["Area Rectangle", "Absolute Value", "Angle Measurement"]
    
    print(f"\n学习路径: {learning_path}")
    
    for i, kc_name in enumerate(learning_path):
        print(f"\n--- Step {i+2}: 学习 '{kc_name}' ---")
        
        result = env.step(student_id, kc_name, concept_map)
        
        assert result['success'], f"Step failed at {kc_name}"
        
        print(f"✅ '{kc_name}' 学习完成")
        print(f"当前掌握度: {result['new_mastery']}")
    
    # 检查最终状态
    state = env.get_state(student_id)
    print(f"\n最终状态:")
    print(f"  - 本轮学习的概念: {[c['kc_name'] for c in state['learned_in_episode']]}")
    print(f"  - 总步数: {state['step_count']}")
    print(f"  - 序列长度: {len(state['skills_seq'])}")
    
    assert state['step_count'] == 4  # 1 (test2) + 3 (test3)
    
    print(f"\n✅ 多步学习测试通过")
    
    return env, students


def test_reward_calculation():
    """测试4: 奖励计算"""
    print("\n" + "="*80)
    print("测试4: 奖励计算")
    print("="*80)
    
    # 创建环境和学生
    env, students = test_multiple_steps()
    
    student_id = students[0]['student_id']
    state = env.get_state(student_id)
    
    print(f"\n学生: {student_id}")
    print(f"初始掌握度: {state['initial_mastery']}")
    print(f"当前掌握度: {state['current_mastery']}")
    
    # 计算奖励
    reward = env.compute_reward(student_id)
    
    print(f"\n终局奖励: {reward:.4f}")
    
    # 计算掌握度提升
    initial_avg = sum(state['initial_mastery'].values()) / len(state['initial_mastery'])
    current_avg = sum(state['current_mastery'].values()) / len(state['current_mastery'])
    absolute_gain = current_avg - initial_avg
    
    print(f"\n掌握度分析:")
    print(f"  - 初始平均掌握度: {initial_avg:.2%}")
    print(f"  - 当前平均掌握度: {current_avg:.2%}")
    print(f"  - 绝对提升: {absolute_gain:.2%}")
    print(f"  - 归一化奖励: {reward:.4f}")
    
    assert 0 <= reward <= 1, f"Reward out of range: {reward}"
    
    print(f"\n✅ 奖励计算正确")


def test_dkt_tool():
    """测试5: DKTUpdateTool"""
    print("\n" + "="*80)
    print("测试5: DKTUpdateTool")
    print("="*80)
    
    # 使用真实的DKT模型路径
    dkt_model_path = "/mnt/hpfs/xiangc/mxy/lpr-r1/DKT/SavedModels/SRC_assist09_kcs_path0_concept138.ckpt"
    
    # 创建环境
    env = LearningPathEnv(
        dkt_model_path=dkt_model_path,
        num_concepts=138
    )
    
    # 创建工具
    tool = DKTUpdateTool(env)
    
    print(f"✅ DKTUpdateTool 创建成功")
    print(f"  - 概念映射: {len(tool.concept_id_map)} 个概念")
    
    # 创建学生数据
    students = create_mock_student_data(2)
    env.reset_batch(students)
    
    # 设置batch上下文
    student_ids = [s['student_id'] for s in students]
    tool.set_batch_context(student_ids)
    
    print(f"\n✅ Batch上下文设置: {student_ids}")
    
    # 测试提取推荐的KC
    llm_output = """
    <think>
    The student needs to learn Ordering Integers first.
    </think>
    <recommend>Ordering Integers</recommend>
    """
    
    extracted_kc = tool.extract_recommended_kc(llm_output)
    print(f"\n从LLM输出提取KC: '{extracted_kc}'")
    assert extracted_kc == "Ordering Integers"
    
    # 测试batch执行（使用真实的ASSIST09概念）
    args_list = [
        {'recommended_kc': 'Ordering Integers', 'student_idx': 0},
        {'recommended_kc': 'Area Rectangle', 'student_idx': 1}
    ]
    
    print(f"\n执行batch更新...")
    results = tool.batch_execute(args_list)
    
    assert len(results) == 2
    
    for i, result in enumerate(results):
        print(f"\n学生 {i} 的观察结果:")
        print(result)
    
    # 测试done检查
    done_output = "<think>Student is ready</think>\n<done/>"
    is_done = tool.check_done(done_output)
    assert is_done
    print(f"\n✅ Done检测: {is_done}")
    
    print(f"\n✅ DKTUpdateTool 测试通过")


def test_batch_processing():
    """测试6: Batch处理"""
    print("\n" + "="*80)
    print("测试6: Batch处理（多个学生）")
    print("="*80)
    
    # 使用真实的DKT模型路径
    dkt_model_path = "/mnt/hpfs/xiangc/mxy/lpr-r1/DKT/SavedModels/SRC_assist09_kcs_path0_concept138.ckpt"
    
    # 创建环境和工具
    env = LearningPathEnv(
        dkt_model_path=dkt_model_path,
        num_concepts=138
    )
    
    tool = DKTUpdateTool(env)
    
    # 创建3个学生
    students = create_mock_student_data(3)
    env.reset_batch(students)
    
    student_ids = [s['student_id'] for s in students]
    tool.set_batch_context(student_ids)
    
    print(f"Batch大小: {len(students)}")
    
    # 模拟多轮推荐（使用真实的ASSIST09概念）
    rounds = [
        ['Ordering Integers', 'Area Rectangle', 'Absolute Value'],
        ['Angle Measurement', 'Ordering Integers', 'Area Rectangle'],
        ['Reflection', 'Absolute Value', 'Ordering Integers']
    ]
    
    for round_num, recommendations in enumerate(rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        args_list = [
            {'recommended_kc': kc, 'student_idx': i}
            for i, kc in enumerate(recommendations)
        ]
        
        results = tool.batch_execute(args_list)
        
        assert len(results) == len(students)
        
        for i, result in enumerate(results):
            print(f"  学生{i}: 学习 '{recommendations[i]}'")
            # print(result[:100] + "...")  # 只打印前100字符
    
    # 检查所有学生的状态
    print(f"\n最终状态:")
    for student_id in student_ids:
        state = env.get_state(student_id)
        reward = env.compute_reward(student_id)
        
        print(f"\n  {student_id}:")
        print(f"    - 学习步数: {state['step_count']}")
        print(f"    - 学习路径: {[c['kc_name'] for c in state['learned_in_episode']]}")
        print(f"    - 终局奖励: {reward:.4f}")
    
    print(f"\n✅ Batch处理测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("DKT Tool 单元测试")
    print("="*80)
    
    try:
        # 测试1: 环境初始化
        test_environment_initialization()
        
        # 测试2: 单步学习
        test_single_step()
        
        # 测试3: 多步学习
        test_multiple_steps()
        
        # 测试4: 奖励计算
        test_reward_calculation()
        
        # 测试5: DKTUpdateTool
        test_dkt_tool()
        
        # 测试6: Batch处理
        test_batch_processing()
        
        print("\n" + "="*80)
        print("✅ 所有测试通过！")
        print("="*80)
        print("\n总结:")
        print("  ✓ 环境初始化")
        print("  ✓ 单步学习")
        print("  ✓ 多步学习路径")
        print("  ✓ 奖励计算")
        print("  ✓ DKTUpdateTool功能")
        print("  ✓ Batch处理")
        print("\nDKT工具系统可以正常工作！")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

