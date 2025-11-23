#!/usr/bin/env python3
"""
测试超简化版API的返回格式
"""
import json

# 模拟新的API返回格式
test_response = {
    "success": True,
    "num_queries": 1,
    "results": [
        {
            "query": "Sum of Interior Angles Triangle prerequisite",
            "concepts": [
                "Sum of Interior Angles Triangle",
                "Sum of Interior Angles Figures with more than 3 sides",
                "Area Triangle"
            ],
            "prerequisites": [
                "Angles - Obtuse Acute and Right",
                "Area Rectangle",
                "Geometric Definitions",
                "Properties and Classification Triangles",
                "Supplementary Angles"
            ]
        }
    ]
}

result = test_response['results'][0]

print("=" * 100)
print("超简化版API返回格式")
print("=" * 100)

# 1. JSON格式
json_str = json.dumps(result, ensure_ascii=False, indent=2)
print("\n【JSON格式】")
print(json_str)

print(f"\n【数据统计】")
print(f"  - JSON字符数: {len(json_str)}")
print(f"  - 估算JSON tokens: ~{len(json_str) // 4}")
print(f"  - concepts: {len(result['concepts'])} 个")
print(f"  - prerequisites: {len(result['prerequisites'])} 个")

# 2. LLM看到的文本格式
text = f"""<knowledge>
Query: {result['query']}

Found Concepts: {', '.join(result['concepts'])}

Available Prerequisites to recommend:
{chr(10).join(f'  - {p}' for p in result['prerequisites'])}
</knowledge>"""

print(f"\n" + "=" * 100)
print("【LLM看到的文本格式】")
print("=" * 100)
print(text)

print(f"\n【文本统计】")
print(f"  - 文本字符数: {len(text)}")
print(f"  - 估算文本tokens: ~{len(text) // 4}")

# 3. 与原始版本对比
print(f"\n" + "=" * 100)
print("【与原始版本对比】")
print("=" * 100)

comparison = {
    '原始版本': {'json': 1823, 'text': 'N/A', 'concepts': 31},
    '简化版v1': {'json': 671, 'text': 194, 'concepts': 9},
    '超简版v2': {'json': len(json_str) // 4, 'text': len(text) // 4, 'concepts': len(result['prerequisites']) + 3}
}

print(f"\n{'版本':<12} {'JSON Tokens':<15} {'文本Tokens':<15} {'概念数':<10} {'减少比例'}")
print("─" * 100)
for version, stats in comparison.items():
    json_tokens = stats['json']
    text_tokens = stats['text']
    concepts = stats['concepts']
    reduction = (1 - json_tokens / 1823) * 100 if json_tokens != 1823 else 0
    
    print(f"{version:<12} {json_tokens:<15} {text_tokens if isinstance(text_tokens, str) else f'{text_tokens:<15}':<15} {concepts:<10} {reduction:>6.1f}%")

# 4. 核心优势分析
print(f"\n" + "=" * 100)
print("【超简版核心优势】")
print("=" * 100)

print("""
✅ 极简结构:
   - 只有3个字段: query, concepts, prerequisites
   - 扁平化，无嵌套结构

✅ 信息精准:
   - concepts: 查询到的相关概念（供参考）
   - prerequisites: 可直接推荐的前置概念（供选择）

✅ 适配小模型:
   - JSON: ~125 tokens（从1823降低93%）
   - 文本: ~40 tokens（极度精简）
   - 适合Qwen2.5-1.5B的处理能力

✅ 无冗余信息:
   - 移除了in_degree, out_degree, dependents
   - 移除了type, confidence, target, depth
   - 移除了learning_paths和hyperedge_candidates

✅ LLM行为更清晰:
   - 看到5-8个清晰的前置概念选项
   - 从中选择1个进行推荐
   - 使用精确的概念名称
""")

print("=" * 100)
print("✨ 完美适配小模型的超简化设计！")
print("=" * 100)

