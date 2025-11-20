#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理脚本：将 ASSIST09 CSV 数据转换为 NPZ 格式
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def preprocess_assist09(csv_path, output_dir, min_interactions=3):
    """
    预处理 ASSIST09 数据集
    
    Args:
        csv_path: CSV 文件路径
        output_dir: 输出目录
        min_interactions: 最小交互次数（过滤掉交互太少的学生）
    """
    print(f"读取数据: {csv_path}")
    df = pd.read_csv(csv_path, encoding='latin1')
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 选择需要的列
    # user_id, problem_id (作为 skill), correct (作为 y)
    df = df[['user_id', 'problem_id', 'correct']].copy()
    
    # 删除缺失值
    df = df.dropna()
    
    # 按 user_id 和 order_id 排序（如果有 order_id 的话）
    # 这里假设数据已经按时间排序
    df = df.sort_values(['user_id'])
    
    print(f"清理后数据形状: {df.shape}")
    print(f"用户数: {df['user_id'].nunique()}")
    print(f"题目数: {df['problem_id'].nunique()}")
    
    # 将 problem_id 重新编号（从 0 开始）
    problem_id_map = {pid: idx for idx, pid in enumerate(sorted(df['problem_id'].unique()))}
    df['skill_id'] = df['problem_id'].map(problem_id_map)
    
    # 按用户分组
    user_groups = df.groupby('user_id')
    
    # 过滤掉交互太少的学生
    valid_users = [user_id for user_id, group in user_groups if len(group) >= min_interactions]
    print(f"过滤后的用户数（至少{min_interactions}次交互）: {len(valid_users)}")
    
    # 构建数据
    skills_list = []
    correct_list = []
    real_len_list = []
    
    print("构建用户序列...")
    for user_id in tqdm(valid_users):
        user_data = df[df['user_id'] == user_id]
        
        # 获取技能序列和答题结果
        skills = user_data['skill_id'].values.astype(np.int32)
        correct = user_data['correct'].values.astype(np.int32)
        
        skills_list.append(skills)
        correct_list.append(correct)
        real_len_list.append(len(skills))
    
    # 转换为 numpy 数组（使用 object 类型，因为每个用户的序列长度不同）
    skills_array = np.array(skills_list, dtype=object)
    correct_array = np.array(correct_list, dtype=object)
    real_len_array = np.array(real_len_list, dtype=np.int32)
    
    # 保存为 NPZ 格式
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'assist09.npz')
    
    print(f"\n保存数据到: {output_path}")
    np.savez(
        output_path,
        skill=skills_array,
        y=correct_array,
        real_len=real_len_array
    )
    
    # 打印统计信息
    print("\n=== 数据统计 ===")
    print(f"用户数: {len(skills_array)}")
    print(f"技能/题目数: {np.max([np.max(s) for s in skills_array]) + 1}")
    print(f"平均序列长度: {np.mean(real_len_array):.2f}")
    print(f"最大序列长度: {np.max(real_len_array)}")
    print(f"最小序列长度: {np.min(real_len_array)}")
    print(f"平均正确率: {np.mean([np.mean(c) for c in correct_array]):.4f}")
    
    # 验证数据
    print("\n验证数据格式...")
    with np.load(output_path, allow_pickle=True) as data:
        print(f"NPZ 文件包含的键: {data.files}")
        print(f"skill 形状: {data['skill'].shape}")
        print(f"y 形状: {data['y'].shape}")
        print(f"real_len 形状: {data['real_len'].shape}")
        print(f"第一个用户的序列长度: {data['real_len'][0]}")
        print(f"第一个用户的前 10 个技能: {data['skill'][0][:10]}")
        print(f"第一个用户的前 10 个答题结果: {data['y'][0][:10]}")
    
    print("\n✅ 数据预处理完成！")


if __name__ == '__main__':
    csv_path = './assistments_2009_2010.csv'
    output_dir = './data/assist09'
    
    # 检查 CSV 文件是否存在
    if not os.path.exists(csv_path):
        print(f"❌ 错误: CSV 文件不存在 - {csv_path}")
        print("请确保 assistments_2009_2010.csv 在当前目录下")
        exit(1)
    
    preprocess_assist09(csv_path, output_dir, min_interactions=3)

