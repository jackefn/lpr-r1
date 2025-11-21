#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试训练好的SRC模型
使用真实的测试集评估模型性能
"""
import os
import sys
import json
import time
from argparse import ArgumentParser, Namespace
import numpy as np
from mindspore import load_param_into_net, load_checkpoint, context, Tensor
import mindspore as ms
from tqdm import tqdm

# 添加父目录到路径以导入KTScripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_src_model(model_path, skill_num=138, embed_size=128, hidden_size=128, dropout=0.3):
    """加载训练好的SRC模型"""
    from Scripts.Agent.SRC import SRC
    
    print(f"\n加载SRC模型:")
    print(f"  模型路径: {model_path}")
    print(f"  Skill数量: {skill_num}")
    print(f"  Embedding维度: {embed_size}")
    print(f"  隐藏层维度: {hidden_size}")
    
    # 创建SRC模型
    model = SRC(
        skill_num=skill_num,
        input_size=embed_size,
        weight_size=embed_size // 2,
        hidden_size=hidden_size,
        dropout=dropout,
        allow_repeat=False,
        with_kt=False
    )
    
    # 加载权重
    if os.path.exists(model_path):
        load_param_into_net(model, load_checkpoint(model_path))
        print(f"  ✅ 模型加载成功\n")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model.set_train(False)
    return model


def load_kt_env(dkt_model_path, skill_num=138):
    """加载KT环境"""
    from kt_env_concepts import KTEnvConcepts
    
    print(f"\n初始化KT环境:")
    print(f"  DKT模型路径: {dkt_model_path}")
    print(f"  KC数量: {skill_num}")
    
    env = KTEnvConcepts(
        model_path=dkt_model_path,
        skill_num=skill_num
    )
    
    return env


def load_test_data(test_data_path):
    """加载测试集"""
    print(f"\n加载测试集:")
    print(f"  数据路径: {test_data_path}")
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"  测试样本数: {len(test_data)}")
    print(f"  ✅ 测试集加载成功\n")
    
    return test_data


def evaluate_student(model, env, student_data, steps=10):
    """
    评估单个学生
    
    Args:
        model: SRC模型
        env: KT环境
        student_data: 学生数据字典
        steps: 学习步数
    
    Returns:
        reward: 学习后的掌握度提升
        initial_mastery: 初始掌握度
        final_mastery: 最终掌握度
        learning_path: 学习路径
    """
    # 提取学生数据
    student_id = student_data['student_id']
    skills_seq = np.array(student_data['skills_seq'], dtype=np.int32)
    corrects_seq = np.array(student_data['corrects_seq'], dtype=np.int32)
    target_concepts = np.array(student_data['target_concepts'], dtype=np.int32)
    
    # 重置环境，使用学生的历史记录初始化
    state = env.reset(
        student_id=student_id,
        targets=target_concepts,
        initial_logs=skills_seq,
        use_train_set=False,
        random_student=False
    )
    
    initial_mastery = state['initial_score']  # 这是一个标量，表示目标KC的平均掌握度
    
    # 准备输入数据（添加batch维度）
    targets = Tensor(target_concepts.reshape(1, -1), ms.int32)
    initial_logs = Tensor(skills_seq[-10:].reshape(1, -1), ms.int32)  # 使用最近10条记录
    
    # 计算初始分数（使用DKT预测）
    initial_log_scores = Tensor([[initial_mastery]], ms.float32)
    
    # 生成参考路径（这里简单使用目标KC重复）
    origin_path = np.tile(target_concepts, (1, steps // len(target_concepts) + 1))[:, :steps]
    origin_path = Tensor(origin_path.reshape(1, steps), ms.int32)
    
    # 使用SRC模型生成学习路径
    with ms._no_grad():
        # model forward: (targets, initial_logs, initial_log_scores, origin_path, steps)
        result = model(targets, initial_logs, initial_log_scores, origin_path, steps)
        # result: (actions, scores, ...)
        learning_path = result[0].asnumpy()[0]  # [steps]
    
    # 模拟学习过程
    for kc_id in learning_path:
        # 执行学习步骤（binary=True表示二元学习）
        state = env.step(
            kc_ids=np.array([int(kc_id)], dtype=np.int32),
            binary=True
        )
    
    # 获取最终掌握度
    final_mastery = state['current_target_score']  # 标量，表示目标KC的平均掌握度
    
    # 计算reward（目标KC上的平均掌握度提升）
    reward = final_mastery - initial_mastery
    
    return {
        'reward': reward,
        'initial_mastery': initial_mastery[target_indices].mean(),
        'final_mastery': final_mastery[target_indices].mean(),
        'learning_path': learning_path.tolist()
    }


def main(args):
    """主测试函数"""
    print("="*80)
    print("SRC模型测试 - 基于真实测试集")
    print("="*80)
    
    # 设置随机种子
    np.random.seed(args.rand_seed)
    
    # 加载模型
    model = load_src_model(
        model_path=args.model_path,
        skill_num=args.skill_num,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout
    )
    
    # 加载KT环境
    env = load_kt_env(
        dkt_model_path=args.dkt_model_path,
        skill_num=args.skill_num
    )
    
    # 加载测试集
    test_data = load_test_data(args.test_data_path)
    
    # 限制测试样本数量
    if args.max_samples > 0 and len(test_data) > args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"⚠️  限制测试样本数为 {args.max_samples}\n")
    
    # 测试
    print("="*80)
    print("开始测试")
    print("="*80 + "\n")
    
    results = []
    failed_count = 0
    
    for i, student_data in enumerate(tqdm(test_data, desc="测试进度")):
        try:
            result = evaluate_student(
                model=model,
                env=env,
                student_data=student_data,
                steps=args.steps
            )
            results.append(result)
            
            # 每100个样本输出一次中间结果
            if (i + 1) % 100 == 0:
                avg_reward = np.mean([r['reward'] for r in results])
                print(f"\n  已测试 {i+1}/{len(test_data)} 样本")
                print(f"  当前平均reward: {avg_reward:.4f}")
        
        except Exception as e:
            failed_count += 1
            if args.verbose:
                print(f"\n⚠️  样本 {i+1} 测试失败: {e}")
            continue
    
    # 统计结果
    print("\n" + "="*80)
    print("测试结果统计")
    print("="*80)
    
    if len(results) == 0:
        print("⚠️  没有成功的测试样本！")
        return
    
    rewards = [r['reward'] for r in results]
    initial_masteries = [r['initial_mastery'] for r in results]
    final_masteries = [r['final_mastery'] for r in results]
    
    print(f"\n成功测试样本数: {len(results)}/{len(test_data)}")
    if failed_count > 0:
        print(f"失败样本数: {failed_count}")
    
    print(f"\n掌握度提升 (Reward):")
    print(f"  平均值: {np.mean(rewards):.4f}")
    print(f"  中位数: {np.median(rewards):.4f}")
    print(f"  标准差: {np.std(rewards):.4f}")
    print(f"  最小值: {np.min(rewards):.4f}")
    print(f"  最大值: {np.max(rewards):.4f}")
    
    print(f"\n初始掌握度:")
    print(f"  平均值: {np.mean(initial_masteries):.4f}")
    
    print(f"\n最终掌握度:")
    print(f"  平均值: {np.mean(final_masteries):.4f}")
    
    print(f"\n提升百分比:")
    improvement = (np.mean(final_masteries) - np.mean(initial_masteries)) / np.mean(initial_masteries) * 100
    print(f"  {improvement:.2f}%")
    
    # 保存详细结果
    if args.save_results:
        output_path = os.path.join(args.output_dir, f'{args.exp_name}_test_results.json')
        os.makedirs(args.output_dir, exist_ok=True)
        
        output_data = {
            'summary': {
                'num_samples': len(results),
                'num_failed': failed_count,
                'avg_reward': float(np.mean(rewards)),
                'median_reward': float(np.median(rewards)),
                'std_reward': float(np.std(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards)),
                'avg_initial_mastery': float(np.mean(initial_masteries)),
                'avg_final_mastery': float(np.mean(final_masteries)),
                'improvement_percentage': float(improvement)
            },
            'details': results[:100] if not args.save_all_details else results  # 默认只保存前100个
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n详细结果已保存至: {output_path}")
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = ArgumentParser("SRC Model Testing - Concept Level")
    
    # 模型配置
    parser.add_argument('--model_path', type=str, 
                       default='./SavedModels/SRC_assist09_kcs_path0_concept138.ckpt',
                       help='SRC模型权重路径')
    parser.add_argument('--dkt_model_path', type=str,
                       default='./SavedModels/DKT_assist09_kcs_concept138.ckpt',
                       help='DKT模型路径')
    parser.add_argument('--test_data_path', type=str,
                       default='./data/assist09/test_set_top3.json',
                       help='测试集路径')
    
    # 模型参数
    parser.add_argument('--skill_num', type=int, default=138,
                       help='KC数量')
    parser.add_argument('--embed_size', type=int, default=128,
                       help='Embedding维度')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout率')
    
    # 测试配置
    parser.add_argument('--steps', type=int, default=10,
                       help='每个episode的学习步数')
    parser.add_argument('--max_samples', type=int, default=-1,
                       help='最大测试样本数（-1表示测试全部）')
    parser.add_argument('--rand_seed', type=int, default=42,
                       help='随机种子')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='./TestResults',
                       help='结果输出目录')
    parser.add_argument('--exp_name', type=str, default='SRC_assist09_kcs_path0_concept138',
                       help='实验名称')
    parser.add_argument('--save_results', action='store_true',
                       help='保存详细结果到JSON文件')
    parser.add_argument('--save_all_details', action='store_true',
                       help='保存所有样本的详细结果（默认只保存前100个）')
    parser.add_argument('--verbose', action='store_true',
                       help='输出详细信息')
    
    # 设备配置
    parser.add_argument('--cuda', type=int, default=-1,
                       help='GPU设备ID，-1表示使用CPU')
    
    args = parser.parse_args()
    
    # 设置运行环境
    if args.cuda >= 0:
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=args.cuda)
        print(f"使用GPU: {args.cuda}")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        print("使用CPU")
    
    main(args)

