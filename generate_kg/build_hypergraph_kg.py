"""
完整的知识超图构建流程
从原始数据集 → 最终超图知识图谱

使用: python build_hypergraph_kg.py --dataset assist09 --data_path ../data/assist09/skill_builder_data_corrected_collapsed.csv
"""
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

class Config:
    def __init__(self, dataset_name, data_path, output_dir):
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API配置
        self.api_key_file = Path(__file__).parent / "openrouter_api_key.txt"
        with open(self.api_key_file, 'r') as f:
            self.api_key = f.read().strip()
        
        self.model = "openai/gpt-4o-mini"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # 参数
        self.batch_size = 30  # 原子KC批次大小
        self.top_k_similar = 10  # 混合策略候选数
        self.max_retries = 3
        self.timeout = 180
        self.knowlp_iterations = 2  # KnowLP迭代次数

# ============================================================================
# 步骤1: 从原始数据提取KC
# ============================================================================

def extract_kcs_from_data(config):
    """从原始CSV提取知识概念"""
    print("="*80)
    print("步骤1: 从原始数据提取知识概念")
    print("="*80)
    
    df = pd.read_csv(config.data_path, encoding='latin1')
    print(f"读取数据: {len(df)} 条记录")
    
    # 检测列名（兼容不同格式）
    if 'list_skills' in df.columns:
        skill_col = 'list_skills'
    elif 'skill_name' in df.columns:
        skill_col = 'skill_name'
    else:
        raise ValueError("找不到技能列，请检查CSV文件格式")
    
    # 过滤有效的skills
    df_filtered = df[df[skill_col].notna()].copy()
    
    # 提取唯一KC名称（处理可能的分号分隔）
    kc_set = set()
    for skills in df_filtered[skill_col].dropna().unique():
        if isinstance(skills, str):
            # 如果包含分号，拆分
            for kc in skills.split(';'):
                kc = kc.strip()
                if kc:
                    kc_set.add(kc)
        else:
            kc_set.add(str(skills))
    
    kc_names = sorted(list(kc_set))
    print(f"✅ 提取 {len(kc_names)} 个唯一KC")
    
    # 保存KC列表
    kc_list_file = config.output_dir / "kc_names.json"
    with open(kc_list_file, 'w', encoding='utf-8') as f:
        json.dump(kc_names, f, indent=2, ensure_ascii=False)
    
    return kc_names

# ============================================================================
# 步骤2: 生成KC解释 (KnowLP)
# ============================================================================

def call_llm(config, messages, temperature=0.1):
    """调用LLM API"""
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature
    }
    
    for attempt in range(config.max_retries):
        try:
            response = requests.post(
                config.api_url, 
                headers=headers, 
                json=data, 
                timeout=config.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < config.max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"  ⚠️ API错误 (尝试 {attempt+1}/{config.max_retries}), 等待{wait_time}秒...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API调用失败: {e}")

def generate_kc_explanations(config, kc_names):
    """为每个KC生成高质量解释 (KnowLP方法)"""
    print("\n" + "="*80)
    print("步骤2: 生成KC解释 (KnowLP)")
    print("="*80)
    
    explanations = {}
    
    for i, kc in enumerate(tqdm(kc_names, desc="生成KC解释"), 1):
        # KnowLP: 生成 -> 评估 -> 改进
        explanation = kc
        
        for iteration in range(config.knowlp_iterations):
            if iteration == 0:
                # 初始生成
                prompt = f"""You are an expert in mathematics education.

Generate a clear, concise explanation for the following knowledge concept:
"{kc}"

The explanation should include:
1. Definition (2-3 sentences)
2. Key components or prerequisites (if any)
3. Example or application context

Keep it under 150 words."""
            else:
                # 改进
                prompt = f"""Improve the following explanation for "{kc}":

Current explanation:
{explanation}

Make it more precise, concise, and educationally sound. Keep under 150 words."""
            
            try:
                explanation = call_llm(config, [{"role": "user", "content": prompt}])
                time.sleep(1)  # 避免限流
            except Exception as e:
                print(f"  ⚠️ KC '{kc}' 解释生成失败: {e}")
                explanation = f"A mathematical concept related to {kc}."
                break
        
        explanations[kc] = explanation.strip()
    
    print(f"✅ 生成 {len(explanations)} 个KC解释")
    
    # 保存
    explanations_file = config.output_dir / "kc_explanations.json"
    with open(explanations_file, 'w', encoding='utf-8') as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)
    
    return explanations

# ============================================================================
# 步骤3: 混合策略提取关系
# ============================================================================

def compute_kc_embeddings(kc_names, explanations):
    """计算KC的TF-IDF嵌入"""
    texts = [explanations.get(kc, kc) for kc in kc_names]
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    embeddings = vectorizer.fit_transform(texts)
    return embeddings

def extract_relations_hybrid(config, kc_names, explanations):
    """混合策略: 余弦相似度预筛选 + LLM判断"""
    print("\n" + "="*80)
    print("步骤3: 混合策略提取先决/相似关系")
    print("="*80)
    
    # 计算embeddings
    print("计算TF-IDF embeddings...")
    embeddings = compute_kc_embeddings(kc_names, explanations)
    
    all_prerequisite = []
    all_similarity = []
    
    for i, source_kc in enumerate(tqdm(kc_names, desc="提取关系")):
        # 计算与所有其他KC的余弦相似度
        source_vec = embeddings[i:i+1]
        similarities = cosine_similarity(source_vec, embeddings)[0]
        
        # 获取top-k候选
        top_indices = np.argsort(similarities)[::-1][1:config.top_k_similar+1]
        candidates = [kc_names[idx] for idx in top_indices]
        
        # 用LLM判断关系
        prompt = f"""Given the source concept: "{source_kc}"
Explanation: {explanations[source_kc][:200]}

And candidate concepts:
{chr(10).join([f'{j+1}. {c}' for j, c in enumerate(candidates)])}

Task: Identify relationships.

Output JSON:
{{
  "prerequisite_relations": [
    {{"target": "concept_name", "confidence": "high/medium"}}
  ],
  "similarity_relations": [
    {{"target": "concept_name"}}
  ]
}}

Rules:
- prerequisite: source is required BEFORE target
- similarity: concepts are related but not prerequisite
- Only include confident relationships
- Use exact concept names from the candidate list"""
        
        try:
            result_str = call_llm(config, [{"role": "user", "content": prompt}])
            
            # 清理JSON
            result_str = result_str.strip()
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0].strip()
            elif "```" in result_str:
                parts = result_str.split("```")
                if len(parts) >= 3:
                    result_str = parts[1].strip()
            
            result = json.loads(result_str)
            
            # 验证并添加
            for rel in result.get("prerequisite_relations", []):
                if rel['target'] in candidates:
                    all_prerequisite.append({
                        'source': source_kc,
                        'target': rel['target'],
                        'confidence': rel.get('confidence', 'medium')
                    })
            
            for rel in result.get("similarity_relations", []):
                if rel['target'] in candidates:
                    all_similarity.append({
                        'source': source_kc,
                        'target': rel['target']
                    })
            
            time.sleep(2)  # 避免限流
            
        except Exception as e:
            print(f"  ⚠️ KC '{source_kc}' 关系提取失败: {e}")
            continue
    
    # 去重
    prereq_dict = {(r['source'], r['target']): r for r in all_prerequisite}
    all_prerequisite = list(prereq_dict.values())
    
    sim_set = set()
    unique_similarity = []
    for rel in all_similarity:
        key = tuple(sorted([rel['source'], rel['target']]))
        if key not in sim_set:
            sim_set.add(key)
            unique_similarity.append(rel)
    
    print(f"✅ 提取先决关系: {len(all_prerequisite)} 条")
    print(f"✅ 提取相似关系: {len(unique_similarity)} 条")
    
    return all_prerequisite, unique_similarity

# ============================================================================
# 步骤4: 拆解组合KC为原子KC
# ============================================================================

def decompose_composite_kcs(kc_names, prerequisite_relations):
    """拆解组合KC为原子KC"""
    print("\n" + "="*80)
    print("步骤4: 拆解组合KC为原子KC")
    print("="*80)
    
    atomic_kcs = set()
    composite_map = {}
    single_kcs = []
    
    for kc in kc_names:
        if ';' in kc:
            atoms = [c.strip() for c in kc.split(';')]
            atomic_kcs.update(atoms)
            composite_map[kc] = atoms
        else:
            atomic_kcs.add(kc)
            single_kcs.append(kc)
    
    atomic_kc_list = sorted(list(atomic_kcs))
    
    print(f"原始概念: {len(kc_names)} 个")
    print(f"原子KC: {len(atomic_kc_list)} 个")
    print(f"组合KC: {len(composite_map)} 个")
    
    # 构建超边
    hyperedges = []
    for rel in prerequisite_relations:
        source = rel['source']
        target = rel['target']
        
        source_atoms = composite_map.get(source, [source]) if ';' in source else [source]
        target_atoms = composite_map.get(target, [target]) if ';' in target else [target]
        
        hyperedges.append({
            'source_concepts': source_atoms,
            'target_concepts': target_atoms,
            'original_source': source,
            'original_target': target
        })
    
    print(f"超边: {len(hyperedges)} 条")
    
    return atomic_kc_list, composite_map, hyperedges

# ============================================================================
# 步骤5: 为原子KC提取先决关系
# ============================================================================

def extract_atomic_relations(config, atomic_kcs):
    """为原子KC提取先决关系"""
    print("\n" + "="*80)
    print("步骤5: 提取原子KC先决关系")
    print("="*80)
    
    batch_size = config.batch_size
    all_relations = []
    
    # 批次提取
    for batch_idx in range(0, len(atomic_kcs), batch_size):
        batch = atomic_kcs[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        total_batches = (len(atomic_kcs) + batch_size - 1) // batch_size
        
        print(f"处理批次 {batch_num}/{total_batches} ({len(batch)}个概念)...")
        
        kc_list_str = "\n".join([f"{i+1}. {kc}" for i, kc in enumerate(batch)])
        
        prompt = f"""You are an expert in mathematics education.

Given these atomic knowledge concepts:

{kc_list_str}

Identify direct prerequisite relationships (A must be learned BEFORE B).

Output JSON:
{{
  "prerequisite_relations": [
    {{"source": "concept_name", "target": "concept_name", "confidence": "high/medium"}},
    ...
  ]
}}

Rules:
- Only DIRECT prerequisites (not transitive)
- Only between concepts IN THIS LIST
- Use exact concept names
- Be conservative"""
        
        try:
            result_str = call_llm(config, [{"role": "user", "content": prompt}])
            
            # 清理JSON
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0].strip()
            elif "```" in result_str:
                parts = result_str.split("```")
                if len(parts) >= 3:
                    result_str = parts[1].strip()
            
            result = json.loads(result_str)
            batch_rels = result.get("prerequisite_relations", [])
            
            # 验证
            valid_rels = []
            for rel in batch_rels:
                if (rel['source'] in batch and rel['target'] in batch 
                    and rel['source'] != rel['target']):
                    valid_rels.append(rel)
            
            all_relations.extend(valid_rels)
            print(f"  ✅ 提取 {len(valid_rels)} 条关系")
            
            time.sleep(2)
            
        except Exception as e:
            print(f"  ❌ 批次 {batch_num} 失败: {e}")
            continue
    
    # 去重
    unique_rels = {}
    for rel in all_relations:
        key = (rel['source'], rel['target'])
        if key not in unique_rels or rel.get('confidence') == 'high':
            unique_rels[key] = rel
    
    final_relations = list(unique_rels.values())
    print(f"✅ 总共提取 {len(final_relations)} 条原子级关系")
    
    return final_relations

# ============================================================================
# 步骤6: 构建超图并展开
# ============================================================================

def build_hypergraph(atomic_kcs, atomic_relations, hyperedges, similarity_relations, 
                     composite_map):
    """构建超图并展开为有向图"""
    print("\n" + "="*80)
    print("步骤6: 构建知识超图")
    print("="*80)
    
    # 超图结构
    hypergraph = {
        'atomic_concepts': atomic_kcs,
        'atomic_prerequisite_relations': atomic_relations,
        'hyperedges': hyperedges,
        'similarity_relations': []
    }
    
    # 转换相似关系为原子级
    atomic_sim = []
    for rel in similarity_relations:
        source = rel['source']
        target = rel['target']
        
        source_atoms = composite_map.get(source, [source]) if ';' in source else [source]
        target_atoms = composite_map.get(target, [target]) if ';' in target else [target]
        
        for s_atom in source_atoms:
            for t_atom in target_atoms:
                if s_atom != t_atom:
                    atomic_sim.append({'source': s_atom, 'target': t_atom})
    
    # 去重
    seen = set()
    unique_sim = []
    for rel in atomic_sim:
        key = tuple(sorted([rel['source'], rel['target']]))
        if key not in seen:
            seen.add(key)
            unique_sim.append(rel)
    
    hypergraph['similarity_relations'] = unique_sim
    
    # 展开超图为有向图
    edges = []
    
    # 添加原子级关系
    for rel in atomic_relations:
        edges.append({
            'source': rel['source'],
            'target': rel['target'],
            'type': 'atomic',
            'confidence': rel.get('confidence', 'medium')
        })
    
    # 展开超边
    for he in hyperedges:
        for s_atom in he['source_concepts']:
            for t_atom in he['target_concepts']:
                if s_atom != t_atom and s_atom in atomic_kcs and t_atom in atomic_kcs:
                    edges.append({
                        'source': s_atom,
                        'target': t_atom,
                        'type': 'hyperedge_expanded'
                    })
    
    # 去重边
    edge_dict = {}
    for edge in edges:
        key = (edge['source'], edge['target'])
        if key not in edge_dict:
            edge_dict[key] = edge
    
    unique_edges = list(edge_dict.values())
    
    # 统计
    from collections import Counter
    in_degrees = Counter()
    out_degrees = Counter()
    for edge in unique_edges:
        out_degrees[edge['source']] += 1
        in_degrees[edge['target']] += 1
    
    root_nodes = [kc for kc in atomic_kcs if in_degrees[kc] == 0]
    leaf_nodes = [kc for kc in atomic_kcs if out_degrees[kc] == 0]
    
    expanded_graph = {
        'concepts': atomic_kcs,
        'prerequisite_relations': unique_edges,
        'statistics': {
            'num_nodes': len(atomic_kcs),
            'num_edges': len(unique_edges),
            'num_root_nodes': len(root_nodes),
            'num_leaf_nodes': len(leaf_nodes)
        }
    }
    
    print(f"✅ 超图节点: {len(atomic_kcs)} 个")
    print(f"✅ 展开边: {len(unique_edges)} 条")
    print(f"✅ 根节点: {len(root_nodes)} 个")
    print(f"✅ 叶子节点: {len(leaf_nodes)} 个")
    
    return hypergraph, expanded_graph

# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='构建知识超图')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (如 assist09)')
    parser.add_argument('--data_path', type=str, required=True, help='原始CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--skip_explanations', action='store_true', help='跳过解释生成（使用已有）')
    
    args = parser.parse_args()
    
    # 配置
    if args.output_dir is None:
        args.output_dir = f"../data/{args.dataset}/kg_output_hypergraph"
    
    config = Config(args.dataset, args.data_path, args.output_dir)
    
    print("\n" + "="*80)
    print("知识超图构建 - 完整流程")
    print("="*80)
    print(f"数据集: {config.dataset_name}")
    print(f"数据路径: {config.data_path}")
    print(f"输出目录: {config.output_dir}")
    print(f"LLM: {config.model}")
    print("="*80 + "\n")
    
    # 步骤1: 提取KC
    kc_names = extract_kcs_from_data(config)
    
    # 步骤2: 生成KC解释
    explanations_file = config.output_dir / "kc_explanations.json"
    if args.skip_explanations and explanations_file.exists():
        print("\n跳过解释生成，加载已有解释...")
        with open(explanations_file, 'r', encoding='utf-8') as f:
            explanations = json.load(f)
    else:
        explanations = generate_kc_explanations(config, kc_names)
    
    # 步骤3: 混合策略提取关系
    prerequisite_rels, similarity_rels = extract_relations_hybrid(
        config, kc_names, explanations
    )
    
    # 步骤4: 拆解组合KC
    atomic_kcs, composite_map, hyperedges = decompose_composite_kcs(
        kc_names, prerequisite_rels
    )
    
    # 步骤5: 提取原子级关系
    atomic_relations = extract_atomic_relations(config, atomic_kcs)
    
    # 步骤6: 构建超图
    hypergraph, expanded_graph = build_hypergraph(
        atomic_kcs, atomic_relations, hyperedges, similarity_rels, composite_map
    )
    
    # 保存结果
    print("\n" + "="*80)
    print("保存结果")
    print("="*80)
    
    hypergraph_file = config.output_dir / "hypergraph.json"
    with open(hypergraph_file, 'w', encoding='utf-8') as f:
        json.dump(hypergraph, f, indent=2, ensure_ascii=False)
    print(f"✅ 超图: {hypergraph_file}")
    
    expanded_file = config.output_dir / "expanded_hypergraph.json"
    with open(expanded_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_graph, f, indent=2, ensure_ascii=False)
    print(f"✅ 展开图: {expanded_file}")
    
    # 保存统计
    stats = {
        'dataset': config.dataset_name,
        'original_concepts': len(kc_names),
        'atomic_concepts': len(atomic_kcs),
        'composite_concepts': len(composite_map),
        'atomic_relations': len(atomic_relations),
        'hyperedges': len(hyperedges),
        'expanded_edges': len(expanded_graph['prerequisite_relations']),
        'root_nodes': expanded_graph['statistics']['num_root_nodes'],
        'leaf_nodes': expanded_graph['statistics']['num_leaf_nodes']
    }
    
    stats_file = config.output_dir / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✅ 统计: {stats_file}")
    
    print("\n" + "="*80)
    print("✅ 知识超图构建完成！")
    print("="*80)
    print(f"\n最终统计:")
    print(f"  原始概念: {stats['original_concepts']} 个")
    print(f"  原子概念: {stats['atomic_concepts']} 个 (减少 {stats['original_concepts'] - stats['atomic_concepts']})")
    print(f"  原子关系: {stats['atomic_relations']} 条")
    print(f"  展开边: {stats['expanded_edges']} 条")
    print(f"  根节点: {stats['root_nodes']} 个")
    print(f"  叶子节点: {stats['leaf_nodes']} 个")
    print(f"\n输出文件位于: {config.output_dir}/")

if __name__ == "__main__":
    main()

