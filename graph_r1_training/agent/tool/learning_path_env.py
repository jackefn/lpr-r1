"""
Learning Path Environment
管理所有学生的DKT状态，提供学习路径推荐的环境接口
"""
import sys
import numpy as np
from typing import Dict, List, Optional
from copy import deepcopy


class LearningPathEnv:
    """
    学习路径推荐环境
    
    职责：
    1. 管理每个学生的DKT状态（skills_seq, corrects_seq）
    2. 调用DKT模型预测掌握度
    3. 模拟学习过程（更新序列）
    4. 计算奖励（掌握度提升）
    """
    
    def __init__(self, dkt_model_path: str, num_concepts: int = 138, dkt_dir: str = '/mnt/hpfs/xiangc/mxy/lpr-r1/DKT'):
        """
        初始化学习路径环境
        
        Args:
            dkt_model_path: DKT模型checkpoint路径
            num_concepts: 概念总数（默认138，与ASSIST09数据集一致）
            dkt_dir: DKT代码目录（用于导入模块）
        """
        self.dkt_model_path = dkt_model_path
        self.num_concepts = num_concepts
        self.dkt_dir = dkt_dir
        
        # 学生状态存储: {student_id: state_dict}
        self.student_states = {}
        
        # 加载DKT模型
        self.dkt_model = self._load_dkt_model()
        
        print(f"✅ LearningPathEnv initialized with {num_concepts} concepts")
    
    def _load_dkt_model(self):
        """加载DKT模型"""
        try:
            import mindspore as ms
            from argparse import Namespace
            
            # 添加DKT目录到路径
            if self.dkt_dir not in sys.path:
                sys.path.insert(0, self.dkt_dir)
            
            from KTScripts.PredictModel import PredictModel
            from KTScripts.utils import load_model
            
            # 设置CPU模式
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
            
            # 模型配置（与训练时一致）
            model_args = Namespace(
                model='DKT',
                feat_nums=self.num_concepts,  # 138个概念（0-137，嵌入层大小=138）
                embed_size=128,
                hidden_size=128,
                pre_hidden_sizes=[256, 64, 16],
                dropout=0.3,
                output_size=1,
                without_label=False
            )
            
            # 创建模型
            model = load_model(model_args)
            
            # 加载权重
            param_dict = ms.load_checkpoint(self.dkt_model_path)
            ms.load_param_into_net(model, param_dict)
            
            # 设置为评估模式
            model.set_train(False)
            
            print(f"✅ DKT model loaded from {self.dkt_model_path}")
            return model
            
        except Exception as e:
            print(f"⚠️ Failed to load DKT model: {e}")
            print("⚠️ Using dummy model for testing")
            return None
    
    def reset_batch(self, batch_data: List[Dict]):
        """
        为一个batch的所有学生初始化状态
        
        Args:
            batch_data: 列表，每个元素包含：
                - student_id: 学生ID
                - skills_seq: 历史练习的KC序列
                - corrects_seq: 历史答题正确性序列
                - target_concepts: 目标概念列表 [{id, name, mastery}, ...]
        """
        self.student_states.clear()
        
        for data in batch_data:
            student_id = data['student_id']
            
            # 深拷贝避免修改原始数据
            self.student_states[student_id] = {
                'skills_seq': list(data.get('skills_seq', [])),
                'corrects_seq': list(data.get('corrects_seq', [])),
                'target_concepts': data.get('target_concepts', []),
                'learned_in_episode': [],  # 本episode学习的KC
                'initial_mastery': {},  # 初始掌握度（用于计算reward）
                'current_mastery': {},  # 当前掌握度
                'step_count': 0
            }
            
            # 计算初始掌握度
            initial_mastery = self._compute_mastery(student_id)
            self.student_states[student_id]['initial_mastery'] = initial_mastery
            self.student_states[student_id]['current_mastery'] = deepcopy(initial_mastery)
        
        print(f"✅ Initialized {len(batch_data)} students")
    
    def step(self, student_id: str, recommended_kc_name: str, concept_id_map: Dict[str, int]) -> Dict:
        """
        执行一步学习：学生学习推荐的KC
        
        Args:
            student_id: 学生ID
            recommended_kc_name: 推荐的KC名称
            concept_id_map: 概念名称到ID的映射
        
        Returns:
            结果字典，包含：
                - success: 是否成功
                - new_mastery: 新的掌握度
                - observation: 观察结果（文本格式）
        """
        if student_id not in self.student_states:
            return {
                'success': False,
                'error': f"Student {student_id} not found in environment",
                'observation': f"Error: Student {student_id} not initialized"
            }
        
        # 获取KC的ID
        if recommended_kc_name not in concept_id_map:
            return {
                'success': False,
                'error': f"Concept '{recommended_kc_name}' not found",
                'observation': f"Error: Unknown concept '{recommended_kc_name}'"
            }
        
        kc_id = concept_id_map[recommended_kc_name]
        state = self.student_states[student_id]
        
        # 预测学习结果（是否能掌握）
        correctness = self._predict_learning_outcome(student_id, kc_id)
        
        # 更新序列
        state['skills_seq'].append(kc_id)
        state['corrects_seq'].append(correctness)
        state['learned_in_episode'].append({
            'kc_name': recommended_kc_name,
            'kc_id': kc_id,
            'correctness': correctness
        })
        state['step_count'] += 1
        
        # 重新计算掌握度
        new_mastery = self._compute_mastery(student_id)
        state['current_mastery'] = new_mastery
        
        # 生成观察结果
        observation = self._format_observation(student_id, recommended_kc_name, correctness, new_mastery)
        
        return {
            'success': True,
            'new_mastery': new_mastery,
            'correctness': correctness,
            'observation': observation
        }
    
    def _predict_learning_outcome(self, student_id: str, kc_id: int) -> int:
        """
        预测学习结果（学生是否能掌握这个KC）
        
        策略：使用DKT预测该KC的掌握度
        - 如果掌握度 > 0.5，认为能掌握 (correctness=1)
        - 否则，认为还需要练习 (correctness=0)
        
        Args:
            student_id: 学生ID
            kc_id: 概念ID
        
        Returns:
            0 或 1
        """
        if self.dkt_model is None:
            # 如果没有DKT模型，使用简单策略
            # 基于历史表现，概率性返回
            state = self.student_states[student_id]
            if state['corrects_seq']:
                overall_accuracy = sum(state['corrects_seq']) / len(state['corrects_seq'])
                return 1 if np.random.random() < overall_accuracy else 0
            return 1  # 默认成功
        
        # 使用DKT预测单个概念的掌握度
        try:
            mastery = self._predict_single_concept(student_id, kc_id)
            return 1 if mastery > 0.5 else 0
        except Exception as e:
            print(f"⚠️ DKT prediction failed: {e}, using default")
            return 1
    
    def _predict_single_concept(self, student_id: str, kc_id: int) -> float:
        """
        使用DKT预测单个概念的掌握度
        
        Args:
            student_id: 学生ID
            kc_id: 概念ID
        
        Returns:
            掌握度 (0-1)
        """
        import mindspore as ms
        from mindspore import Tensor
        
        state = self.student_states[student_id]
        skills = state['skills_seq']
        corrects = state['corrects_seq']
        
        if not skills:
            return 0.3  # 默认低掌握度
        
        # 构造输入
        x = np.array([skills], dtype=np.int32)  # [1, seq_len]
        y = np.array([corrects], dtype=np.float32)  # [1, seq_len]
        
        x_tensor = Tensor(x, ms.int32)
        y_tensor = Tensor(y, ms.float32)
        
        # 使用learn方法获取预测
        predictions, _ = self.dkt_model.learn(x_tensor)
        
        # 如果KC在序列中出现过，用最后一次的预测
        kc_indices = [i for i, s in enumerate(skills) if s == kc_id]
        if kc_indices:
            last_idx = kc_indices[-1]
            mastery = float(predictions[0, last_idx].asnumpy())
        else:
            # 未练习过，使用整体平均
            mastery = float(predictions[0, -1].asnumpy()) * 0.6
        
        return max(0.0, min(1.0, mastery))
    
    def _compute_mastery(self, student_id: str) -> Dict[str, float]:
        """
        计算学生对所有目标概念的掌握度
        
        Args:
            student_id: 学生ID
        
        Returns:
            {concept_name: mastery_score, ...}
        """
        state = self.student_states[student_id]
        target_concepts = state['target_concepts']
        
        mastery = {}
        for concept in target_concepts:
            kc_id = concept['id']
            kc_name = concept['name']
            
            if self.dkt_model is None:
                # 简化方法：基于该概念的历史正确率
                skills = state['skills_seq']
                corrects = state['corrects_seq']
                kc_indices = [i for i, s in enumerate(skills) if s == kc_id]
                
                if kc_indices:
                    kc_corrects = [corrects[i] for i in kc_indices]
                    mastery[kc_name] = sum(kc_corrects) / len(kc_corrects)
                else:
                    # 未练习过，低掌握度
                    overall = sum(corrects) / len(corrects) if corrects else 0.5
                    mastery[kc_name] = overall * 0.6
            else:
                # 使用DKT模型
                try:
                    mastery[kc_name] = self._predict_single_concept(student_id, kc_id)
                except:
                    mastery[kc_name] = 0.3
        
        return mastery
    
    def _format_observation(self, student_id: str, kc_name: str, correctness: int, new_mastery: Dict) -> str:
        """
        格式化观察结果为LLM可读的文本
        
        Args:
            student_id: 学生ID
            kc_name: 学习的KC名称
            correctness: 学习结果 (0/1)
            new_mastery: 新的掌握度
        
        Returns:
            格式化的观察文本
        """
        state = self.student_states[student_id]
        
        # 构建观察文本
        lines = ["<observation>"]
        lines.append(f"The student has learned '{kc_name}'.")
        
        # 学习结果
        if correctness == 1:
            lines.append(f"Learning outcome: Successfully mastered.")
        else:
            lines.append(f"Learning outcome: Needs more practice.")
        
        # 更新后的掌握度
        lines.append("")
        lines.append("Updated mastery levels:")
        for concept_name, mastery_score in new_mastery.items():
            percentage = int(mastery_score * 100)
            lines.append(f"- {concept_name}: {percentage}%")
        
        # 本轮已学习的概念
        learned = state['learned_in_episode']
        if learned:
            lines.append("")
            lines.append(f"Concepts learned so far: {[c['kc_name'] for c in learned]}")
        
        lines.append("</observation>")
        
        return "\n".join(lines)
    
    def get_state(self, student_id: str) -> Optional[Dict]:
        """获取学生当前状态"""
        return self.student_states.get(student_id)
    
    def compute_reward(self, student_id: str) -> float:
        """
        计算终局奖励（掌握度提升）
        
        Reward = (E_final - E_start) / (1.0 - E_start)
        
        Args:
            student_id: 学生ID
        
        Returns:
            奖励值
        """
        state = self.student_states.get(student_id)
        if not state:
            return 0.0
        
        initial = state['initial_mastery']
        current = state['current_mastery']
        
        # 对所有目标概念求平均
        initial_avg = sum(initial.values()) / len(initial) if initial else 0.0
        current_avg = sum(current.values()) / len(current) if current else 0.0
        
        # 归一化奖励
        if initial_avg >= 0.99:
            return 0.0  # 已经完全掌握
        
        gain = current_avg - initial_avg
        if gain < 0:
            return -0.1  # 小惩罚
        
        normalized_gain = gain / (1.0 - initial_avg)
        return max(0.0, min(1.0, normalized_gain))

