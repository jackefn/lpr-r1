"""
构建 ASSIST09 知识超图的 FAISS 索引
从 expanded_hypergraph.json 生成：
1. index_entity.bin - 概念实体索引
2. index_hyperedge.bin - 先决关系（超边）索引
3. kv_store_entities.json - 实体元数据
4. kv_store_hyperedges.json - 超边元数据
"""
import json
import numpy as np
import faiss
from FlagEmbedding import FlagAutoModel
from tqdm import tqdm
import os

print("="*80)
print("ASSIST09 知识超图 FAISS 索引构建")
print("="*80)

# ============================================================================
# 1. 配置
# ============================================================================
HYPERGRAPH_PATH = "../data/assist09/kg_output_hypergraph/expanded_hypergraph.json"
OUTPUT_DIR = "expr/assist09_hypergraph"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 2. 加载嵌入模型
# ============================================================================
print("\n[1/6] 加载嵌入模型...")
model = FlagAutoModel.from_finetuned(
    EMBEDDING_MODEL,
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    devices="cpu",  # 如果有GPU，改为 "cuda"
)
print(f"✅ 模型加载完成: {EMBEDDING_MODEL}")

# ============================================================================
# 3. 加载知识超图
# ============================================================================
print("\n[2/6] 加载知识超图...")
with open(HYPERGRAPH_PATH, 'r', encoding='utf-8') as f:
    hypergraph = json.load(f)

concepts = hypergraph['concepts']
relations = hypergraph['prerequisite_relations']
stats = hypergraph['statistics']

print(f"✅ 超图加载完成:")
print(f"  - 概念数: {stats['num_nodes']}")
print(f"  - 关系数: {stats['num_edges']}")

# ============================================================================
# 4. 构建实体（概念）索引
# ============================================================================
print("\n[3/6] 构建概念实体索引...")

# 4.1 为每个概念创建描述文本
entity_descriptions = []
kv_store_entities = {}

for idx, concept in enumerate(tqdm(concepts, desc="生成概念描述")):
    # 创建丰富的概念描述（用于更好的语义匹配）
    description = f"Knowledge Concept: {concept}. " \
                  f"This is a mathematical or educational concept related to learning and problem-solving."
    entity_descriptions.append(description)
    
    # 保存到 KV store
    kv_store_entities[str(idx)] = {
        'entity_id': str(idx),
        'entity_name': concept,
        'entity_type': 'concept',
        'description': description
    }

# 4.2 生成嵌入
print("生成概念嵌入向量...")
entity_embeddings = model.encode(entity_descriptions)
entity_embeddings = np.array(entity_embeddings).astype('float32')

# 4.3 创建 FAISS 索引
print("创建 FAISS 索引...")
dimension = entity_embeddings.shape[1]
index_entity = faiss.IndexFlatIP(dimension)  # Inner Product (适合归一化向量)
faiss.normalize_L2(entity_embeddings)  # L2 归一化
index_entity.add(entity_embeddings)

# 4.4 保存
entity_index_path = os.path.join(OUTPUT_DIR, 'index_entity.bin')
entity_kv_path = os.path.join(OUTPUT_DIR, 'kv_store_entities.json')

faiss.write_index(index_entity, entity_index_path)
with open(entity_kv_path, 'w', encoding='utf-8') as f:
    json.dump(kv_store_entities, f, indent=2, ensure_ascii=False)

print(f"✅ 实体索引已保存:")
print(f"  - {entity_index_path}")
print(f"  - {entity_kv_path}")
print(f"  - 向量维度: {dimension}")
print(f"  - 索引大小: {index_entity.ntotal}")

# ============================================================================
# 5. 构建超边（先决关系）索引
# ============================================================================
print("\n[4/6] 构建先决关系（超边）索引...")

# 5.1 为每条关系创建描述文本
hyperedge_descriptions = []
kv_store_hyperedges = {}

for idx, rel in enumerate(tqdm(relations, desc="生成关系描述")):
    source = rel['source']
    target = rel['target']
    rel_type = rel.get('type', 'prerequisite')
    confidence = rel.get('confidence', 'medium')
    
    # 创建关系描述
    if rel_type == 'atomic':
        description = f"Prerequisite Relation: '{source}' is a prerequisite for learning '{target}'. " \
                      f"Students should master {source} before studying {target}. " \
                      f"Confidence: {confidence}."
    else:  # hyperedge_expanded
        description = f"Learning Path: '{source}' leads to '{target}' through hyperedge expansion. " \
                      f"This represents a composite prerequisite relationship."
    
    hyperedge_descriptions.append(description)
    
    # 保存到 KV store
    kv_store_hyperedges[str(idx)] = {
        'hyperedge_id': str(idx),
        'source_entity': source,
        'target_entity': target,
        'relation_type': rel_type,
        'confidence': confidence,
        'content': description,
        'source_id': str(concepts.index(source)),
        'target_id': str(concepts.index(target))
    }

# 5.2 生成嵌入
print("生成关系嵌入向量...")
hyperedge_embeddings = model.encode(hyperedge_descriptions)
hyperedge_embeddings = np.array(hyperedge_embeddings).astype('float32')

# 5.3 创建 FAISS 索引
print("创建 FAISS 索引...")
index_hyperedge = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(hyperedge_embeddings)
index_hyperedge.add(hyperedge_embeddings)

# 5.4 保存
hyperedge_index_path = os.path.join(OUTPUT_DIR, 'index_hyperedge.bin')
hyperedge_kv_path = os.path.join(OUTPUT_DIR, 'kv_store_hyperedges.json')

faiss.write_index(index_hyperedge, hyperedge_index_path)
with open(hyperedge_kv_path, 'w', encoding='utf-8') as f:
    json.dump(kv_store_hyperedges, f, indent=2, ensure_ascii=False)

print(f"✅ 超边索引已保存:")
print(f"  - {hyperedge_index_path}")
print(f"  - {hyperedge_kv_path}")
print(f"  - 向量维度: {dimension}")
print(f"  - 索引大小: {index_hyperedge.ntotal}")

# ============================================================================
# 6. 保存图谱元数据
# ============================================================================
print("\n[5/6] 保存图谱元数据...")

metadata = {
    'dataset': 'assist09_hypergraph',
    'num_concepts': len(concepts),
    'num_relations': len(relations),
    'embedding_model': EMBEDDING_MODEL,
    'embedding_dimension': int(dimension),
    'index_entity_size': int(index_entity.ntotal),
    'index_hyperedge_size': int(index_hyperedge.ntotal),
    'relation_types': {
        'atomic': sum(1 for r in relations if r.get('type') == 'atomic'),
        'hyperedge_expanded': sum(1 for r in relations if r.get('type') == 'hyperedge_expanded')
    }
}

metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ 元数据已保存: {metadata_path}")

# ============================================================================
# 7. 测试索引
# ============================================================================
print("\n[6/6] 测试索引...")

test_queries = [
    "What should I learn before studying quadratic equations?",
    "Prerequisites for algebra",
    "Basic concepts for geometry"
]

print("\n测试查询结果:")
for query in test_queries:
    print(f"\n🔍 Query: {query}")
    
    # 查询嵌入
    query_embedding = model.encode_queries([query])
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # 检索实体
    _, entity_ids = index_entity.search(query_embedding, 3)
    print(f"  Top 3 相关概念:")
    for i, eid in enumerate(entity_ids[0], 1):
        print(f"    {i}. {concepts[eid]}")
    
    # 检索超边
    _, hyperedge_ids = index_hyperedge.search(query_embedding, 3)
    print(f"  Top 3 相关关系:")
    for i, hid in enumerate(hyperedge_ids[0], 1):
        rel = relations[hid]
        print(f"    {i}. {rel['source']} → {rel['target']} ({rel.get('type', 'unknown')})")

# ============================================================================
# 完成
# ============================================================================
print("\n" + "="*80)
print("✅ ASSIST09 知识超图索引构建完成！")
print("="*80)
print(f"\n生成的文件:")
print(f"  📁 {OUTPUT_DIR}/")
print(f"    ├── index_entity.bin          ({os.path.getsize(entity_index_path) / 1024 / 1024:.2f} MB)")
print(f"    ├── index_hyperedge.bin       ({os.path.getsize(hyperedge_index_path) / 1024 / 1024:.2f} MB)")
print(f"    ├── kv_store_entities.json    ({os.path.getsize(entity_kv_path) / 1024:.2f} KB)")
print(f"    ├── kv_store_hyperedges.json  ({os.path.getsize(hyperedge_kv_path) / 1024:.2f} KB)")
print(f"    └── metadata.json             ({os.path.getsize(metadata_path) / 1024:.2f} KB)")

print(f"\n下一步:")
print(f"  运行 API 服务: python script_api_assist09.py")

