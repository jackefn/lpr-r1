#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版SRC模型测试脚本
直接使用与训练相同的环境和数据格式
"""
import os
import sys
import json
import numpy as np
from argparse import ArgumentParser, Namespace
from mindspore import load_param_into_net, load_checkpoint, context
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config_from_model_name(model_name):
    """从模型名称中解析配置"""
    parts = model_name.split('_')
    config = {
        'agent': parts[0],  # SRC
        'dataset': '_'.join(parts[1:-2]),  # assist09_kcs
        'path': int(parts[-2].replace('path', '')),  # 0
        'concept_num': int(parts[-1].replace('concept', '').replace('.ckpt', ''))  # 138
    }
    return config


def main(args):
    """主测试函数"""
    print("="*80)
    print("SRC模型测试 - 简化版")
    print("="*80)
    
    # 从模型名称解析配置
    model_name = os.path.basename(args.model_path)
    config = load_config_from_model_name(model_name)
    
    print(f"\n模型配置:")
    print(f"  Agent: {config['agent']}")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Path类型: {config['path']}")
    print(f"  Concept数量: {config['concept_num']}")
    print(f"  模型路径: {args.model_path}")
    print(f"  DKT路径: {args.dkt_model_path}\n")
    
    # 设置随机种子
    from KTScripts.utils import set_random_seed
    set_random_seed(args.rand_seed)
    
    # 导入必要的模块
    from trainSRC_concepts import ConceptKESEnv, load_agent, get_concept_data
    
    # 创建环境
    print("初始化环境...")
    from KTScripts.DataLoader import KTDataset
    # KTDataset接受data_folder参数，它会提取basename作为folder_name
    # 然后加载 data_folder/folder_name.npz
    # 所以传入 './data/assist09_kcs'，它会加载 './data/assist09_kcs/assist09_kcs.npz'
    data_path = os.path.join(args.data_dir, config['dataset'])
    dataset = KTDataset(data_path)
    env = ConceptKESEnv(dataset, args.dkt_model_path, skill_num=config['concept_num'])
    
    # 加载模型
    print("加载SRC模型...")
    args_for_agent = Namespace(
        agent=config['agent'],
        skill_num=config['concept_num'],
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout
    )
    model = load_agent(args_for_agent)
    
    if os.path.exists(args.model_path):
        load_param_into_net(model, load_checkpoint(args.model_path))
        print(f"  ✅ 模型加载成功\n")
    else:
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    model.set_train(False)
    
    # 开始测试
    print("="*80)
    print(f"开始测试（测试{args.test_iterations}个batch）")
    print("="*80 + "\n")
    
    test_rewards = []
    
    for i in tqdm(range(args.test_iterations), desc="测试进度"):
        # 获取测试数据
        targets, initial_logs, origin_path = get_concept_data(
            args.batch_size, config['concept_num'],
            args.num_targets, args.num_initial_logs,
            config['path'], args.steps
        )
        
        # 开始episode
        initial_log_scores = env.begin_episode(targets, initial_logs)
        data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
        
        # 模型推理
        result = model(*data)
        
        # 执行学习路径
        env.n_step(result[0], binary=True)
        
        # 获取reward
        rewards = env.end_episode()
        
        mean_reward = np.mean(rewards.asnumpy())
        test_rewards.append(mean_reward)
        
        # 每50个batch打印一次
        if (i + 1) % 50 == 0 or i == 0:
            current_avg = np.mean(test_rewards)
            print(f'  Batch {i+1}/{args.test_iterations} | '
                  f'当前reward: {mean_reward:.4f} | '
                  f'平均reward: {current_avg:.4f}')
    
    # 统计结果
    print("\n" + "="*80)
    print("测试结果统计")
    print("="*80)
    
    print(f"\n总测试Batch数: {len(test_rewards)}")
    print(f"每个Batch大小: {args.batch_size}")
    print(f"总测试样本数: {len(test_rewards) * args.batch_size}")
    
    print(f"\nReward统计:")
    print(f"  平均值: {np.mean(test_rewards):.4f}")
    print(f"  中位数: {np.median(test_rewards):.4f}")
    print(f"  标准差: {np.std(test_rewards):.4f}")
    print(f"  最小值: {np.min(test_rewards):.4f}")
    print(f"  最大值: {np.max(test_rewards):.4f}")
    
    # 保存结果
    if args.save_results:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'model_config': config,
            'test_config': {
                'batch_size': args.batch_size,
                'test_iterations': args.test_iterations,
                'total_samples': len(test_rewards) * args.batch_size,
                'steps': args.steps,
                'num_targets': args.num_targets,
                'num_initial_logs': args.num_initial_logs
            },
            'statistics': {
                'mean_reward': float(np.mean(test_rewards)),
                'median_reward': float(np.median(test_rewards)),
                'std_reward': float(np.std(test_rewards)),
                'min_reward': float(np.min(test_rewards)),
                'max_reward': float(np.max(test_rewards))
            },
            'all_rewards': [float(r) for r in test_rewards]
        }
        
        output_path = os.path.join(output_dir, f'{model_name.replace(".ckpt", "")}_test_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ 详细结果已保存至: {output_path}")
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = ArgumentParser("SRC Model Testing - Simple Version")
    
    # 模型配置
    parser.add_argument('--model_path', type=str,
                       default='./SavedModels/SRC_assist09_kcs_path0_concept138.ckpt',
                       help='SRC模型路径')
    parser.add_argument('--dkt_model_path', type=str,
                       default='./SavedModels/DKT_assist09_kcs_concept138.ckpt',
                       help='DKT模型路径')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    
    # 模型参数（需要与训练时一致）
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # 测试配置
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--test_iterations', type=int, default=200,
                       help='测试迭代次数')
    parser.add_argument('--steps', type=int, default=10,
                       help='每个episode的学习步数')
    parser.add_argument('--num_targets', type=int, default=3,
                       help='目标KC数量')
    parser.add_argument('--num_initial_logs', type=int, default=10,
                       help='初始学习记录数量')
    parser.add_argument('--rand_seed', type=int, default=42)
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='./TestResults')
    parser.add_argument('--save_results', action='store_true',
                       help='保存测试结果')
    
    # 设备配置
    parser.add_argument('--cuda', type=int, default=-1)
    
    args = parser.parse_args()
    
    # 设置运行环境
    if args.cuda >= 0:
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=args.cuda)
        print(f"使用GPU: {args.cuda}")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        print("使用CPU")
    
    main(args)

