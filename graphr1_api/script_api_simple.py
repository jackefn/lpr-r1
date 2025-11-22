"""
ASSIST09 知识超图 API 服务 - 简化版
直接使用 FAISS + 知识图谱遍历，不依赖 GraphR1
"""
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import faiss
from FlagEmbedding import FlagAutoModel
from typing import List, Dict, Optional
import argparse
import os
import networkx as nx

# ============================================================================
# 配置
# ============================================================================
parser = argparse.ArgumentParser(description='ASSIST09 知识超图 API 服务 - 简化版')
parser.add_argument('--data_source', default='assist09_hypergraph', help='数据源名称')
parser.add_argument('--port', type=int, default=8002, help='API 服务端口')
parser.add_argument('--host', default='0.0.0.0', help='API 服务地址')
args = parser.parse_args()

data_source = args.data_source
DATA_DIR = f"expr/{data_source}"
HYPERGRAPH_PATH = "../data/assist09/kg_output_hypergraph/expanded_hypergraph.json"

print("="*80)
print(f"ASSIST09 知识超图 API 启动中（简化版）...")
print("="*80)

# ============================================================================
# 加载嵌入模型
# ============================================================================
print("\n[1/5] 加载嵌入模型...")
model = FlagAutoModel.from_finetuned(
    'BAAI/bge-large-en-v1.5',
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    devices="cpu",
)
print("✅ 嵌入模型加载完成")

# ============================================================================
# 加载实体索引
# ============================================================================
print("\n[2/5] 加载实体（概念）索引...")
entity_index_path = os.path.join(DATA_DIR, 'index_entity.bin')
entity_kv_path = os.path.join(DATA_DIR, 'kv_store_entities.json')

index_entity = faiss.read_index(entity_index_path)
with open(entity_kv_path, 'r', encoding='utf-8') as f:
    entities = json.load(f)

corpus_entity = []
for item in entities:
    corpus_entity.append(entities[item]['entity_name'])

print(f"✅ 实体索引加载完成: {len(corpus_entity)} 个概念")

# ============================================================================
# 加载超边索引
# ============================================================================
print("\n[3/5] 加载超边（关系）索引...")
hyperedge_index_path = os.path.join(DATA_DIR, 'index_hyperedge.bin')
hyperedge_kv_path = os.path.join(DATA_DIR, 'kv_store_hyperedges.json')

index_hyperedge = faiss.read_index(hyperedge_index_path)
with open(hyperedge_kv_path, 'r', encoding='utf-8') as f:
    hyperedges = json.load(f)

corpus_hyperedge = []
hyperedge_map = {}  # source -> target 映射
for item in hyperedges:
    content = hyperedges[item]['content']
    corpus_hyperedge.append(content)
    source = hyperedges[item]['source_entity']
    target = hyperedges[item]['target_entity']
    if source not in hyperedge_map:
        hyperedge_map[source] = []
    hyperedge_map[source].append(target)

print(f"✅ 超边索引加载完成: {len(corpus_hyperedge)} 条关系")

# ============================================================================
# 加载知识图谱
# ============================================================================
print("\n[4/5] 加载知识图谱...")
with open(HYPERGRAPH_PATH, 'r', encoding='utf-8') as f:
    hypergraph_data = json.load(f)

# 构建 NetworkX 图
G = nx.DiGraph()
G.add_nodes_from(hypergraph_data['concepts'])
for rel in hypergraph_data['prerequisite_relations']:
    G.add_edge(rel['source'], rel['target'], **rel)

print(f"✅ 知识图谱加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# ============================================================================
# 核心功能函数
# ============================================================================
def find_prerequisites(concept: str, max_depth: int = 3) -> List[Dict]:
    """找到一个概念的所有前置概念"""
    if concept not in G:
        return []
    
    prerequisites = []
    visited = set()
    
    def dfs(node, depth):
        if depth > max_depth or node in visited:
            return
        visited.add(node)
        
        for pred in G.predecessors(node):
            edge_data = G[pred][node]
            prerequisites.append({
                'prerequisite': pred,
                'target': node,
                'depth': depth,
                'type': edge_data.get('type', 'unknown'),
                'confidence': edge_data.get('confidence', 'medium')
            })
            dfs(pred, depth + 1)
    
    dfs(concept, 0)
    return prerequisites

def find_learning_path(start_concepts: List[str], target_concept: str) -> List[List[str]]:
    """找到从起点概念到目标概念的学习路径"""
    paths = []
    for start in start_concepts:
        if start in G and target_concept in G:
            if nx.has_path(G, start, target_concept):
                try:
                    path = nx.shortest_path(G, start, target_concept)
                    paths.append(path)
                except:
                    pass
    return paths

def get_concept_info(concept: str) -> Dict:
    """获取概念的详细信息"""
    if concept not in G:
        return None
    
    return {
        'name': concept,
        'in_degree': G.in_degree(concept),
        'out_degree': G.out_degree(concept),
        'prerequisites': list(G.predecessors(concept)),
        'dependents': list(G.successors(concept))
    }

# ============================================================================
# 查询处理
# ============================================================================
async def process_queries(
    queries: List[str],
    top_k_retrieval: int = 5,
    max_depth: int = 3
) -> List[Dict]:
    """处理查询并返回结果"""
    
    # 1. FAISS 检索
    embeddings = model.encode_queries(queries)
    _, entity_ids = index_entity.search(embeddings, top_k_retrieval)
    _, hyperedge_ids = index_hyperedge.search(embeddings, top_k_retrieval)
    
    results = []
    for i, query in enumerate(queries):
        # 检索到的实体
        entities_found = [corpus_entity[idx] for idx in entity_ids[i] if idx < len(corpus_entity)]
        
        # 检索到的关系
        hyperedges_found = [corpus_hyperedge[idx] for idx in hyperedge_ids[i] if idx < len(corpus_hyperedge)]
        
        # 对每个实体找前置概念
        all_prerequisites = []
        learning_paths = []
        concept_details = []
        
        for entity in entities_found[:3]:  # 只处理 top 3
            # 获取概念信息
            info = get_concept_info(entity)
            if info:
                concept_details.append(info)
            
            # 找前置概念
            prereqs = find_prerequisites(entity, max_depth=max_depth)
            all_prerequisites.extend(prereqs)
            
            # 找学习路径（从根节点到该概念）
            root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
            paths = find_learning_path(root_nodes[:5], entity)
            learning_paths.extend(paths)
        
        # 构建结果
        results.append({
            'query': query,
            'entity_candidates': entities_found,
            'hyperedge_candidates': hyperedges_found,
            'concept_details': concept_details,
            'prerequisites': all_prerequisites[:20],  # 限制数量
            'learning_paths': [{'path': p, 'length': len(p)-1} for p in learning_paths[:5]],
            'num_prerequisites': len(all_prerequisites),
            'num_paths': len(learning_paths)
        })
    
    return results

# ============================================================================
# FastAPI 应用
# ============================================================================
app = FastAPI(
    title="ASSIST09 Knowledge Hypergraph API (Simplified)",
    description="简化版 ASSIST09 知识超图检索 API",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    queries: List[str]
    top_k_retrieval: Optional[int] = 5
    max_depth: Optional[int] = 3

class SearchResponse(BaseModel):
    success: bool
    num_queries: int
    results: List[Dict]

@app.get("/")
async def root():
    return {
        "service": "ASSIST09 Knowledge Hypergraph API (Simplified)",
        "version": "1.0.0",
        "num_concepts": len(corpus_entity),
        "num_relations": len(corpus_hyperedge),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "running"}

@app.get("/concept/{concept_name}")
async def get_concept(concept_name: str):
    """获取单个概念的详细信息"""
    info = get_concept_info(concept_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"概念 '{concept_name}' 不存在")
    
    prereqs = find_prerequisites(concept_name, max_depth=5)
    return {
        **info,
        'all_prerequisites': prereqs
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """搜索接口"""
    try:
        if not request.queries:
            raise HTTPException(status_code=400, detail="查询列表不能为空")
        
        results = await process_queries(
            request.queries,
            top_k_retrieval=request.top_k_retrieval,
            max_depth=request.max_depth
        )
        
        return SearchResponse(
            success=True,
            num_queries=len(request.queries),
            results=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理查询时出错: {str(e)}")

# ============================================================================
# 启动服务
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 启动 API 服务（简化版）...")
    print("="*80)
    print(f"地址: http://{args.host}:{args.port}")
    print(f"文档: http://{args.host}:{args.port}/docs")
    print("="*80 + "\n")
    
    uvicorn.run(app, host=args.host, port=args.port)

