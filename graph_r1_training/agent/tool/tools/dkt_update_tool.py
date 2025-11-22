"""
DKT Update Tool
处理LLM推荐的知识点，更新学生的DKT状态和掌握度
"""
import re
from typing import Dict, List
from agent.tool.tool_base import Tool
from agent.tool.learning_path_env import LearningPathEnv


class DKTUpdateTool(Tool):
    """
    DKT更新工具
    
    功能：
    1. 从LLM输出中提取推荐的KC
    2. 调用LearningPathEnv更新学生状态
    3. 返回观察结果（文本格式）
    
    注意：
    - 此Tool需要配合LearningPathEnv使用
    - 需要在RewardManager中传递学生上下文
    """
    
    def __init__(self, env: LearningPathEnv):
        """
        初始化DKT更新工具
        
        Args:
            env: LearningPathEnv实例
        """
        name = "dkt_update"
        description = "Update student's knowledge state after learning a recommended concept."
        parameters = {
            "type": "object",
            "properties": {
                "recommended_kc": {
                    "type": "string",
                    "description": "The recommended knowledge concept name"
                }
            },
            "required": ["recommended_kc"]
        }
        
        super().__init__(name, description, parameters)
        self.env = env
        
        # 概念名称到ID的映射（需要从环境或数据加载）
        self.concept_id_map = self._load_concept_map()
        
        # 当前batch的学生ID列表（在reset时设置）
        self.current_batch_student_ids = []
    
    def _load_concept_map(self) -> Dict[str, int]:
        """
        加载概念名称到ID的映射
        
        从知识图谱文件加载
        """
        try:
            import json
            kg_path = '/mnt/hpfs/xiangc/mxy/lpr-r1/data/assist09/kg_output_hypergraph/expanded_hypergraph.json'
            
            with open(kg_path, 'r') as f:
                kg = json.load(f)
            
            concept_map = {name: i for i, name in enumerate(kg['concepts'])}
            print(f"✅ Loaded {len(concept_map)} concepts for DKTUpdateTool")
            return concept_map
            
        except Exception as e:
            print(f"⚠️ Failed to load concept map: {e}")
            return {}
    
    def set_batch_context(self, student_ids: List[str]):
        """
        设置当前batch的学生ID列表
        
        在每个batch开始时调用
        
        Args:
            student_ids: 学生ID列表（与batch中的顺序对应）
        """
        self.current_batch_student_ids = student_ids
    
    def execute(self, args: Dict) -> str:
        """
        单次执行（不推荐使用，应使用batch_execute）
        
        Args:
            args: 参数字典，包含recommended_kc
        
        Returns:
            观察结果文本
        """
        # 不支持单次执行
        return "Error: DKTUpdateTool only supports batch execution"
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        批量执行：为多个学生更新DKT状态
        
        注意：虽然叫batch_execute，但实际是逐个学生顺序处理
        
        Args:
            args_list: 参数列表，每个元素包含：
                - recommended_kc: 推荐的KC名称
                - student_idx: 学生在batch中的索引（由RewardManager添加）
        
        Returns:
            观察结果列表（文本格式）
        """
        results = []
        
        for i, args in enumerate(args_list):
            # 获取推荐的KC
            recommended_kc = args.get('recommended_kc', '')
            
            # 获取学生ID
            if 'student_idx' in args:
                student_idx = args['student_idx']
                if student_idx < len(self.current_batch_student_ids):
                    student_id = self.current_batch_student_ids[student_idx]
                else:
                    results.append(f"Error: Invalid student index {student_idx}")
                    continue
            else:
                # 如果没有student_idx，使用顺序索引
                if i < len(self.current_batch_student_ids):
                    student_id = self.current_batch_student_ids[i]
                else:
                    results.append(f"Error: No student context for index {i}")
                    continue
            
            # 执行更新
            result = self.env.step(student_id, recommended_kc, self.concept_id_map)
            
            if result['success']:
                results.append(result['observation'])
            else:
                error_msg = result.get('error', 'Unknown error')
                results.append(f"<observation>\nError: {error_msg}\n</observation>")
        
        return results
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        计算即时奖励
        
        对于DKT工具，即时奖励为0，终局奖励在RewardManager中计算
        
        Args:
            args: 工具参数
            result: 工具执行结果
        
        Returns:
            0.0 (终局奖励)
        """
        return 0.0
    
    @staticmethod
    def extract_recommended_kc(text: str) -> str:
        """
        从LLM输出中提取推荐的KC
        
        查找 <recommend>...</recommend> 标签
        
        Args:
            text: LLM生成的文本
        
        Returns:
            推荐的KC名称，如果未找到则返回空字符串
        """
        pattern = r'<recommend>(.*?)</recommend>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            kc_name = match.group(1).strip()
            return kc_name
        
        return ""
    
    @staticmethod
    def check_done(text: str) -> bool:
        """
        检查LLM是否输出了<done/>标签
        
        Args:
            text: LLM生成的文本
        
        Returns:
            True如果包含<done/>
        """
        return '<done/>' in text or '<done>' in text

