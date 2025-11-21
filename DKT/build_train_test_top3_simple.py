#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 ASSIST09 的训练集和测试集
从每个学生的未练习概念中，选择最不常见的3个作为目标
（作为"掌握度最低"的代理指标）
"""
import os
import numpy as np
import json
from tqdm import tqdm
import random
from collections import Counter


def load_assist09_data(data_path):
    """加载 assist09_kcs.npz 数据"""
    data = np.load(data_path, allow_pickle=True)
    skills = data['skill']
    corrects = data['y']
    real_len = data['real_len']
    skill_num = int(data['skill_num'])
    
    print(f"加载数据:")
    print(f"  学生数: {len(skills)}")
    print(f"  知识概念数: {skill_num}")
    
    return skills, corrects, real_len, skill_num


def compute_kc_frequencies(skills):
    """
    计算每个KC在整个数据集中的出现频率
    频率越低 -> 越不常见 -> 越难/越少人掌握
    """
    kc_counter = Counter()
    
    for student_skills in skills:
        kc_counter.update(student_skills)
    
    total_occurrences = sum(kc_counter.values())
    kc_freq = {kc: count / total_occurrences for kc, count in kc_counter.items()}
    
    return kc_freq


def select_top3_rarest_targets(practiced_kcs, all_kcs, kc_freq, top_k=3):
    """
    从未练习的概念中，选择最不常见的top_k个作为目标
    
    Args:
        practiced_kcs: 学生已练习的KC集合
        all_kcs: 所有KC的集合
        kc_freq: KC频率字典
        top_k: 选择的数量
        
    Returns:
        selected_targets: 选中的目标KC列表（按频率升序）
        target_frequencies: 对应的频率
    """
    candidate_kcs = list(all_kcs - practiced_kcs)
    
    if len(candidate_kcs) == 0:
        return [], []
    
    # 按频率排序（升序，频率越低越靠前）
    # 对于未在数据集中出现的KC，频率为0
    sorted_candidates = sorted(candidate_kcs, key=lambda kc: kc_freq.get(kc, 0))
    
    # 选择前top_k个（最不常见的）
    selected = sorted_candidates[:min(top_k, len(sorted_candidates))]
    frequencies = [kc_freq.get(kc, 0) for kc in selected]
    
    return selected, frequencies


def build_dataset(skills, corrects, real_len, skill_num, kc_freq,
                 min_history=5, target_count=3):
    """
    构建数据集
    """
    all_kcs = set(range(skill_num))
    dataset = []
    
    print(f"\n开始处理学生数据...")
    print(f"  最小历史记录要求: {min_history}")
    print(f"  每学生目标概念数: {target_count}")
    print(f"  策略: 选择最不常见的{target_count}个未练习KC")
    
    for student_id in tqdm(range(len(skills))):
        skills_seq = skills[student_id]
        corrects_seq = corrects[student_id]
        
        # 检查序列长度
        if len(skills_seq) < min_history or len(skills_seq) != len(corrects_seq):
            continue
        
        practiced_kcs = set(skills_seq)
        
        # 选择最不常见的目标概念
        selected_targets, target_freqs = select_top3_rarest_targets(
            practiced_kcs, all_kcs, kc_freq, top_k=target_count
        )
        
        if len(selected_targets) >= 1:  # 至少有1个目标
            dataset.append({
                'student_id': int(student_id),
                'skills_seq': [int(x) for x in skills_seq],
                'corrects_seq': [int(x) for x in corrects_seq],
                'target_concepts': [int(x) for x in selected_targets],
                'target_frequencies': [float(x) for x in target_freqs],
                'num_practiced': len(practiced_kcs),
                'num_targets': len(selected_targets),
                'seq_length': len(skills_seq)
            })
    
    return dataset


def split_dataset(dataset, train_ratio=0.6):
    """划分训练集和测试集"""
    random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    return dataset[:split_idx], dataset[split_idx:]


def compute_statistics(dataset, name="Dataset"):
    """计算数据集统计信息"""
    if len(dataset) == 0:
        print(f"\n{name}: 空数据集")
        return
    
    num_students = len(dataset)
    total_targets = sum(sample['num_targets'] for sample in dataset)
    avg_targets_per_student = total_targets / num_students
    
    total_practiced = sum(sample['num_practiced'] for sample in dataset)
    avg_practiced_per_student = total_practiced / num_students
    
    # 序列长度统计
    seq_lengths = [sample['seq_length'] for sample in dataset]
    avg_seq_length = np.mean(seq_lengths)
    
    # 目标KC频率统计
    all_freqs = []
    for sample in dataset:
        all_freqs.extend(sample['target_frequencies'])
    avg_freq = np.mean(all_freqs) if len(all_freqs) > 0 else 0
    max_freq = max(all_freqs) if len(all_freqs) > 0 else 0
    min_freq = min(all_freqs) if len(all_freqs) > 0 else 0
    
    print(f"\n{name} 统计:")
    print(f"  学生数: {num_students}")
    print(f"  总目标概念数: {total_targets}")
    print(f"  平均每学生的目标概念数: {avg_targets_per_student:.2f}")
    print(f"  平均每学生已练习概念数: {avg_practiced_per_student:.2f}")
    print(f"  平均序列长度: {avg_seq_length:.2f}")
    print(f"  目标概念频率（越低越难）:")
    print(f"    - 平均: {avg_freq:.6f}")
    print(f"    - 最小: {min_freq:.6f}")
    print(f"    - 最大: {max_freq:.6f}")


def save_dataset(train_set, test_set, output_dir, skill_num, kc_freq):
    """保存数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_set_top3.json')
    test_path = os.path.join(output_dir, 'test_set_top3.json')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_set, f, indent=2)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=2)
    
    metadata = {
        'skill_num': int(skill_num),
        'train_size': int(len(train_set)),
        'test_size': int(len(test_set)),
        'total_size': int(len(train_set) + len(test_set)),
        'train_ratio': float(len(train_set) / (len(train_set) + len(test_set))),
        'targets_per_student': 3,
        'selection_strategy': 'rarest_kcs_in_dataset',
        'strategy_description': '选择数据集中最不常见的3个未练习KC作为目标（越不常见越难）'
    }
    
    metadata_path = os.path.join(output_dir, 'dataset_metadata_top3.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # 保存KC频率信息
    kc_freq_path = os.path.join(output_dir, 'kc_frequencies.json')
    kc_freq_sorted = {str(k): float(v) for k, v in sorted(kc_freq.items(), key=lambda x: x[1])}
    with open(kc_freq_path, 'w', encoding='utf-8') as f:
        json.dump(kc_freq_sorted, f, indent=2)
    
    print(f"\n数据集已保存:")
    print(f"  训练集: {train_path}")
    print(f"  测试集: {test_path}")
    print(f"  元数据: {metadata_path}")
    print(f"  KC频率: {kc_freq_path}")


def main():
    # 配置
    data_path = '/mnt/hpfs/xiangc/mxy/lpr-r1/DKT/data/assist09/assist09_kcs.npz'
    output_dir = '/mnt/hpfs/xiangc/mxy/lpr-r1/DKT/data/assist09'
    train_ratio = 0.6
    min_history = 5
    target_count = 3  # 每个学生选择3个目标
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    print("=" * 60)
    print("ASSIST09 训练集/测试集构建（Top-3目标概念）")
    print("=" * 60)
    print(f"配置:")
    print(f"  数据路径: {data_path}")
    print(f"  训练集比例: {train_ratio}")
    print(f"  测试集比例: {1 - train_ratio}")
    print(f"  最小历史记录: {min_history}")
    print(f"  每学生目标数: {target_count}")
    print(f"\n策略:")
    print(f"  - 计算每个KC在数据集中的出现频率")
    print(f"  - 对每个学生，从未练习的KC中选择最不常见的{target_count}个")
    print(f"  - 理由: 不常见的KC通常更难，掌握的人更少")
    print(f"  - 这是'初始掌握度最低'的合理代理指标")
    
    # 1. 加载数据
    skills, corrects, real_len, skill_num = load_assist09_data(data_path)
    
    # 2. 计算KC频率
    print("\n计算KC频率...")
    kc_freq = compute_kc_frequencies(skills)
    print(f"  总共 {len(kc_freq)} 个KC有记录")
    
    # 显示最不常见的10个KC
    sorted_kcs = sorted(kc_freq.items(), key=lambda x: x[1])
    print(f"\n  最不常见的10个KC（作为目标优先级最高）:")
    for i, (kc, freq) in enumerate(sorted_kcs[:10], 1):
        print(f"    {i}. KC_{kc}: 频率={freq:.6f}")
    
    # 3. 构建数据集
    dataset = build_dataset(
        skills, corrects, real_len, skill_num, kc_freq,
        min_history=min_history,
        target_count=target_count
    )
    
    print(f"\n总共构建 {len(dataset)} 个有效样本")
    
    if len(dataset) == 0:
        print("错误: 没有符合条件的样本！")
        return
    
    # 4. 划分训练集和测试集
    print("\n划分训练集和测试集...")
    train_set, test_set = split_dataset(dataset, train_ratio)
    
    # 5. 统计信息
    compute_statistics(dataset, "完整数据集")
    compute_statistics(train_set, "训练集")
    compute_statistics(test_set, "测试集")
    
    # 6. 保存数据集
    save_dataset(train_set, test_set, output_dir, skill_num, kc_freq)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print("\n说明:")
    print("  - 当前DKT模型需要使用新的KC数据重新训练")
    print("  - 在此之前，使用KC频率作为难度的代理指标")
    print("  - 频率越低的KC，通常意味着：")
    print("    * 在课程中出现较少（更高级或更专业）")
    print("    * 学生接触较少（初始掌握度更低）")
    print("    * 可以作为合理的学习目标")


if __name__ == '__main__':
    main()

