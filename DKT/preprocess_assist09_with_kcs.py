#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理脚本：将 ASSIST09 CSV 数据转换为 NPZ 格式
使用 list_skill_ids 字段作为知识概念（KCs）
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def parse_skill_ids(skill_ids_str):
    """
    解析 list_skill_ids 字段
    例如: "10;12" -> [10, 12]
          "279" -> [279]
    """
    if pd.isna(skill_ids_str):
        return []
    
    try:
        # 分割并转换为整数
        skills = [int(s.strip()) for s in str(skill_ids_str).split(';') if s.strip()]
        return skills
    except:
        return []


def preprocess_assist09_with_kcs(csv_path, output_dir, min_interactions=5, use_first_skill_only=True):
    """
    预处理 ASSIST09 数据集，使用 list_skill_ids 作为知识概念
    
    Args:
        csv_path: CSV 文件路径
        output_dir: 输出目录
        min_interactions: 最小交互次数（过滤掉交互太少的学生）
        use_first_skill_only: 对于多技能的题目，是否只使用第一个技能
    """
    print(f"读取数据: {csv_path}")
    df = pd.read_csv(csv_path, encoding='latin1')
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 选择需要的列
    df = df[['user_id', 'problem_id', 'correct', 'list_skill_ids']].copy()
    
    # 删除缺失的 correct 值
    df = df[df['correct'].notna()].copy()
    
    # 只保留有 list_skill_ids 的行
    print(f"\n过滤前: {len(df)} 行")
    df = df[df['list_skill_ids'].notna()].copy()
    print(f"过滤后（只保留有 list_skill_ids 的行）: {len(df)} 行")
    
    # 解析 list_skill_ids
    print("\n解析 list_skill_ids...")
    df['parsed_skills'] = df['list_skill_ids'].apply(parse_skill_ids)
    
    # 移除无法解析的行
    df = df[df['parsed_skills'].apply(len) > 0].copy()
    print(f"解析后: {len(df)} 行")
    
    # 对于多技能的题目，选择策略
    if use_first_skill_only:
        print("\n策略: 使用第一个技能ID")
        df['skill_id'] = df['parsed_skills'].apply(lambda x: x[0])
    else:
        # 展开：一个题目如果有多个技能，变成多行
        print("\n策略: 展开多技能（一个题目一行变为多行）")
        rows = []
        for _, row in df.iterrows():
            for skill in row['parsed_skills']:
                rows.append({
                    'user_id': row['user_id'],
                    'problem_id': row['problem_id'],
                    'correct': row['correct'],
                    'skill_id': skill
                })
        df = pd.DataFrame(rows)
    
    print(f"\n最终数据形状: {df.shape}")
    print(f"用户数: {df['user_id'].nunique()}")
    print(f"题目数: {df['problem_id'].nunique()}")
    print(f"唯一技能数: {df['skill_id'].nunique()}")
    print(f"技能ID范围: {df['skill_id'].min()} - {df['skill_id'].max()}")
    
    # 将 skill_id 重新编号（从 0 开始）
    print("\n重新编号技能ID...")
    unique_skills = sorted(df['skill_id'].unique())
    skill_id_map = {sid: idx for idx, sid in enumerate(unique_skills)}
    df['skill_id_mapped'] = df['skill_id'].map(skill_id_map)
    
    print(f"映射后技能数: {len(unique_skills)}")
    print(f"映射后技能ID范围: 0 - {len(unique_skills) - 1}")
    
    # 按用户分组
    user_groups = df.groupby('user_id')
    
    # 过滤掉交互太少的学生
    valid_users = [user_id for user_id, group in user_groups if len(group) >= min_interactions]
    print(f"\n过滤后的用户数（至少{min_interactions}次交互）: {len(valid_users)}")
    
    # 构建数据
    skills_list = []
    correct_list = []
    real_len_list = []
    
    print("\n构建用户序列...")
    for user_id in tqdm(valid_users):
        user_data = df[df['user_id'] == user_id].sort_values('problem_id')  # 按题目ID排序保持顺序
        
        # 获取技能序列和答题结果
        skills = user_data['skill_id_mapped'].values.astype(np.int32)
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
    output_path = os.path.join(output_dir, 'assist09_kcs.npz')
    
    print(f"\n保存数据到: {output_path}")
    np.savez(
        output_path,
        skill=skills_array,
        y=correct_array,
        real_len=real_len_array,
        skill_num=len(unique_skills)
    )
    
    # 保存技能ID映射
    skill_map_path = os.path.join(output_dir, 'skill_id_mapping.txt')
    with open(skill_map_path, 'w') as f:
        f.write("mapped_id\toriginal_id\n")
        for orig_id, mapped_id in sorted(skill_id_map.items(), key=lambda x: x[1]):
            f.write(f"{mapped_id}\t{orig_id}\n")
    print(f"技能ID映射保存到: {skill_map_path}")
    
    # 统计信息
    print(f"\n最终统计:")
    print(f"  学生数: {len(valid_users)}")
    print(f"  知识概念数: {len(unique_skills)}")
    print(f"  总交互记录数: {sum(real_len_list)}")
    print(f"  平均每学生交互数: {sum(real_len_list) / len(valid_users):.2f}")
    print(f"  正确率: {sum([correct_array[i].sum() for i in range(len(correct_array))]) / sum(real_len_list):.4f}")
    
    print("\n完成!")


def main():
    csv_path = '/mnt/hpfs/xiangc/mxy/lpr-r1/data/assist09/assistments_2009_2010.csv'
    output_dir = '/mnt/hpfs/xiangc/mxy/lpr-r1/DKT/data/assist09'
    
    preprocess_assist09_with_kcs(
        csv_path=csv_path,
        output_dir=output_dir,
        min_interactions=5,
        use_first_skill_only=True  # 对于多技能题目，只使用第一个技能
    )


if __name__ == '__main__':
    main()

