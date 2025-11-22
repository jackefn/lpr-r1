"""
测试 ASSIST09 知识超图 API
"""
import requests
import json
from typing import List, Dict

# ============================================================================
# 配置
# ============================================================================
API_BASE_URL = "http://localhost:8001"

# ============================================================================
# 测试用例
# ============================================================================
TEST_QUERIES = [
    "What should I learn before studying quadratic equations?",
    "Prerequisites for algebra",
    "Basic concepts for geometry",
    "What knowledge is needed for solving linear equations?",
    "How to prepare for learning fractions?",
]

# ============================================================================
# 测试函数
# ============================================================================
def test_health():
    """测试健康检查接口"""
    print("\n" + "="*80)
    print("测试 1: 健康检查")
    print("="*80)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ 健康检查通过")
        print(f"响应: {json.dumps(result, indent=2)}")
        return True
    
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_info():
    """测试信息接口"""
    print("\n" + "="*80)
    print("测试 2: 服务信息")
    print("="*80)
    
    try:
        response = requests.get(f"{API_BASE_URL}/info")
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ 信息获取成功")
        print(f"数据源: {result['data_source']}")
        print(f"概念数: {result['num_concepts']}")
        print(f"关系数: {result['num_relations']}")
        print(f"嵌入模型: {result['embedding_model']}")
        return True
    
    except Exception as e:
        print(f"❌ 信息获取失败: {e}")
        return False

def test_search_single():
    """测试单个查询"""
    print("\n" + "="*80)
    print("测试 3: 单个查询")
    print("="*80)
    
    query = TEST_QUERIES[0]
    print(f"\n🔍 查询: {query}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "queries": [query],
                "top_k_retrieval": 5,
                "top_k_final": 10
            }
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"\n✅ 查询成功")
        print(f"成功: {result['success']}")
        print(f"查询数: {result['num_queries']}")
        
        if result['results']:
            first_result = result['results'][0]
            print(f"\n📊 结果详情:")
            print(f"  查询: {first_result['query']}")
            print(f"\n  Top 5 相关概念:")
            for i, entity in enumerate(first_result['entity_candidates'], 1):
                print(f"    {i}. {entity}")
            
            print(f"\n  Top 5 相关关系:")
            for i, hyperedge in enumerate(first_result['hyperedge_candidates'][:5], 1):
                print(f"    {i}. {hyperedge}")
        
        return True
    
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        if hasattr(e, 'response'):
            print(f"响应内容: {e.response.text}")
        return False

def test_search_batch():
    """测试批量查询"""
    print("\n" + "="*80)
    print("测试 4: 批量查询")
    print("="*80)
    
    queries = TEST_QUERIES[:3]
    print(f"\n📝 批量查询 {len(queries)} 个问题:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "queries": queries,
                "top_k_retrieval": 3,
                "top_k_final": 5
            }
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"\n✅ 批量查询成功")
        print(f"成功: {result['success']}")
        print(f"查询数: {result['num_queries']}")
        
        print(f"\n📊 各查询结果摘要:")
        for i, res in enumerate(result['results'], 1):
            print(f"\n  查询 {i}: {res['query'][:50]}...")
            print(f"    相关概念: {', '.join(res['entity_candidates'][:3])}")
        
        return True
    
    except Exception as e:
        print(f"❌ 批量查询失败: {e}")
        if hasattr(e, 'response'):
            print(f"响应内容: {e.response.text}")
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*80)
    print("测试 5: 边界情况")
    print("="*80)
    
    test_cases = [
        {
            "name": "空查询列表",
            "payload": {"queries": []},
            "should_fail": True
        },
        {
            "name": "超大 top_k",
            "payload": {"queries": ["test"], "top_k_retrieval": 100, "top_k_final": 100},
            "should_fail": False
        },
        {
            "name": "非英文查询",
            "payload": {"queries": ["什么是代数？"]},
            "should_fail": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        try:
            response = requests.post(
                f"{API_BASE_URL}/search",
                json=test_case['payload']
            )
            
            if test_case['should_fail']:
                if response.status_code >= 400:
                    print(f"  ✅ 按预期失败 (状态码: {response.status_code})")
                else:
                    print(f"  ⚠️ 应该失败但成功了")
            else:
                response.raise_for_status()
                print(f"  ✅ 测试通过")
        
        except Exception as e:
            if test_case['should_fail']:
                print(f"  ✅ 按预期失败: {type(e).__name__}")
            else:
                print(f"  ❌ 意外失败: {e}")
    
    return True

# ============================================================================
# 主测试流程
# ============================================================================
def main():
    """运行所有测试"""
    print("="*80)
    print("ASSIST09 知识超图 API 测试套件")
    print("="*80)
    print(f"API 地址: {API_BASE_URL}")
    
    # 检查服务是否运行
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=2)
    except:
        print("\n❌ 错误: API 服务未运行")
        print(f"请先启动服务: python script_api_assist09.py")
        return
    
    # 运行测试
    results = []
    results.append(("健康检查", test_health()))
    results.append(("服务信息", test_info()))
    results.append(("单个查询", test_search_single()))
    results.append(("批量查询", test_search_batch()))
    results.append(("边界情况", test_edge_cases()))
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print("\n" + "="*80)
    print(f"总计: {passed}/{total} 通过")
    print("="*80)
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败")

if __name__ == "__main__":
    main()

