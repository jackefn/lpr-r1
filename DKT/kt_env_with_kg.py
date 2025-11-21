"""
KT环境与知识图谱集成示例

展示如何：
1. 从知识图谱中采样相关的目标KCs
2. 使用逐步奖励
3. 为Graph-R1准备好的接口
"""
import json
import numpy as np
from typing import List, Tuple, Dict
from kt_env import KTEnv


class KTEnvWithKG(KTEnv):
    """
    扩展KTEnv，集成知识图谱功能
    """
    
    def __init__(self, kg_path: str = None, *args, **kwargs):
        """
        初始化环境
        
        Args:
            kg_path: 知识图谱JSON文件路径
        """
        super().__init__(*args, **kwargs)
        
        # 加载知识图谱
        self.kg = None
        self.kc_to_id = {}  # KC名称 → ID映射
        self.id_to_kc = {}  # ID → KC名称映射
        
        if kg_path:
            self.load_kg(kg_path)
        
        # 用于逐步奖励
        self.last_score = None
        self.step_count = 0
    
    def load_kg(self, kg_path: str):
        """加载知识图谱"""
        with open(kg_path, 'r', encoding='utf-8') as f:
            self.kg = json.load(f)
        
        print(f"✅ 加载知识图谱:")
        print(f"   - 概念数: {len(self.kg['concepts'])}")
        print(f"   - 前置关系: {len(self.kg.get('prerequisite_relations', []))}")
        print(f"   - 相似关系: {len(self.kg.get('similarity_relations', []))}")
        
        # 构建映射（假设KG中的concept名称可以映射到skill ID）
        # 这里需要根据实际数据调整
        # 例如：如果concept名称是skill名称，需要一个skill_name → skill_id的映射
    
    def sample_targets_from_kg(self, num_targets: int = 3, strategy: str = 'connected') -> np.ndarray:
        """
        从知识图谱中采样目标KCs
        
        Args:
            num_targets: 目标数量
            strategy: 采样策略
                - 'random': 随机采样
                - 'connected': 采样相连的KCs（学习单元）
                - 'prerequisite_chain': 采样前置链
        
        Returns:
            targets: shape (1, num_targets)
        """
        if self.kg is None or strategy == 'random':
            # 如果没有KG或选择随机，使用随机采样
            targets = np.random.choice(self.skill_num, num_targets, replace=False)
            return targets.reshape(1, -1)
        
        if strategy == 'connected':
            return self._sample_connected_targets(num_targets)
        elif strategy == 'prerequisite_chain':
            return self._sample_prerequisite_chain(num_targets)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _sample_connected_targets(self, num_targets: int) -> np.ndarray:
        """
        采样相连的KCs作为学习单元
        
        策略：
        1. 随机选一个source KC
        2. 通过图谱找它的neighbors（前置+相似）
        3. 组合成一个learning unit
        """
        if not self.kg:
            return self.sample_targets_from_kg(num_targets, strategy='random')
        
        # 构建邻接表
        neighbors = {i: set() for i in range(self.skill_num)}
        
        # 添加前置关系的邻居
        for rel in self.kg.get('prerequisite_relations', []):
            # 需要将concept名称映射到ID
            # 这里简化处理，假设已有映射
            source_id = self._concept_to_id(rel['source'])
            target_id = self._concept_to_id(rel['target'])
            if source_id is not None and target_id is not None:
                neighbors[source_id].add(target_id)
                neighbors[target_id].add(source_id)
        
        # 添加相似关系的邻居
        for rel in self.kg.get('similarity_relations', []):
            source_id = self._concept_to_id(rel['source'])
            target_id = self._concept_to_id(rel['target'])
            if source_id is not None and target_id is not None:
                neighbors[source_id].add(target_id)
                neighbors[target_id].add(source_id)
        
        # 随机选一个source
        source = np.random.randint(0, self.skill_num)
        
        # BFS找邻居
        candidates = list(neighbors[source])
        
        # 如果邻居不够，随机补充
        if len(candidates) < num_targets - 1:
            remaining = set(range(self.skill_num)) - set(candidates) - {source}
            additional = np.random.choice(list(remaining), 
                                         num_targets - 1 - len(candidates), 
                                         replace=False)
            candidates.extend(additional)
        
        # 采样num_targets-1个邻居
        if len(candidates) > num_targets - 1:
            neighbors_sample = np.random.choice(candidates, num_targets - 1, replace=False)
        else:
            neighbors_sample = candidates
        
        # 组合source和neighbors
        targets = [source] + list(neighbors_sample)
        return np.array([targets])
    
    def _sample_prerequisite_chain(self, num_targets: int) -> np.ndarray:
        """
        采样前置链作为学习路径
        
        例如：[KC_A → KC_B → KC_C]
        学生需要按顺序掌握
        """
        # TODO: 实现前置链采样
        # 这需要对KG进行拓扑排序，找到一条长度为num_targets的路径
        return self.sample_targets_from_kg(num_targets, strategy='random')
    
    def _concept_to_id(self, concept_name: str) -> int:
        """
        将concept名称映射到skill ID
        
        这需要根据实际数据调整
        """
        # 简化实现：随机映射（实际需要正确的映射表）
        if concept_name in self.kc_to_id:
            return self.kc_to_id[concept_name]
        else:
            # 如果没有映射，返回None
            return None
    
    def reset(self, targets=None, initial_logs=None, use_kg=True, num_targets=3):
        """
        重置环境（扩展版）
        
        Args:
            targets: 目标KCs（如果提供则使用，否则从KG采样）
            initial_logs: 初始学习记录
            use_kg: 是否使用KG采样目标
            num_targets: 目标数量（当targets为None时）
        
        Returns:
            state_info: 状态信息
        """
        # 如果没有提供targets，则从KG采样
        if targets is None and use_kg and self.kg is not None:
            targets = self.sample_targets_from_kg(num_targets, strategy='connected')
        elif targets is None:
            targets = np.random.choice(self.skill_num, num_targets, replace=False).reshape(1, -1)
        
        # 调用父类reset
        state_info = super().reset(targets, initial_logs)
        
        # 初始化逐步奖励相关变量
        self.last_score = self.initial_score.copy()
        self.step_count = 0
        
        return state_info
    
    def step(self, kc_ids, binary=True, return_step_reward=True):
        """
        执行一步（扩展版）
        
        Args:
            kc_ids: 要学习的KCs
            binary: 是否二值化
            return_step_reward: 是否返回逐步奖励
        
        Returns:
            step_info: 包含逐步奖励的信息
        """
        # 调用父类step
        step_info = super().step(kc_ids, binary)
        
        # 计算逐步奖励
        if return_step_reward:
            step_reward = self.current_score - self.last_score
            step_info['step_reward'] = step_reward.asnumpy()
            
            # 更新last_score
            self.last_score = self.current_score.copy()
        
        # 更新step计数
        self.step_count += 1
        
        # 判断是否完成（可选）
        mastery_threshold = 0.8
        done = bool((self.current_score >= mastery_threshold).asnumpy().all())
        step_info['done'] = done
        
        # 添加诊断信息
        step_info['info'] = {
            'step_count': self.step_count,
            'target_progress': self.current_score.asnumpy().tolist(),
            'mastery_gain': step_reward.asnumpy().tolist() if return_step_reward else None
        }
        
        return step_info


def demo_kg_sampling():
    """演示如何使用KG采样目标"""
    print("=" * 80)
    print("Demo: KT环境 + 知识图谱集成")
    print("=" * 80)
    
    # 创建环境（暂时不加载KG，因为需要正确的映射）
    env = KTEnvWithKG(
        kg_path=None,  # 如果有KG，提供路径
        model_name='DKT',
        dataset_name='assist09'
    )
    
    print("\n--- 场景1: 使用KG采样相连的KCs作为目标 ---")
    # 如果有KG，会采样相连的KCs
    # 否则随机采样
    state = env.reset(use_kg=False, num_targets=3)  # 这里暂时用随机
    
    print(f"目标KCs: {env.targets.asnumpy()[0]}")
    print(f"初始掌握度: {state['initial_score'][0]:.4f}")
    
    print("\n--- 场景2: 逐步奖励 ---")
    # 执行学习路径
    learning_path = np.random.randint(0, env.skill_num, (1, 10))
    
    total_reward = 0
    for i in range(10):
        kc = learning_path[:, i:i+1]
        step_info = env.step(kc, return_step_reward=True)
        
        step_reward = step_info['step_reward'][0]
        total_reward += step_reward
        
        if i % 3 == 0:
            print(f"Step {i}: 学习KC {kc[0,0]}, "
                  f"逐步奖励={step_reward:.4f}, "
                  f"目标掌握度={step_info['current_target_score'][0]:.4f}")
    
    print(f"\n总逐步奖励: {total_reward:.4f}")
    
    # 终局奖励
    final_reward = env.get_reward(full_score=3)[0]
    print(f"终局奖励: {final_reward:.4f}")
    
    print("\n✅ Demo完成")


def demo_graph_r1_interface():
    """
    演示Graph-R1使用接口
    
    这展示了如何在RL训练循环中使用KTEnv
    """
    print("\n" + "=" * 80)
    print("Demo: Graph-R1 RL训练接口")
    print("=" * 80)
    
    env = KTEnvWithKG(model_name='DKT', dataset_name='assist09')
    
    # 模拟RL训练循环
    num_episodes = 3
    max_steps = 20
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        # 1. 重置环境（采样新目标）
        state = env.reset(use_kg=False, num_targets=3)
        targets = env.targets.asnumpy()[0]
        
        print(f"目标: {targets}")
        print(f"初始掌握度: {state['initial_score'][0]:.4f}")
        
        # 2. RL循环
        episode_reward = 0
        for step in range(max_steps):
            # [这里Graph-R1策略会选择action]
            # 简化：随机选择
            action = np.random.randint(0, env.skill_num, (1, 1))
            
            # 3. 环境step
            step_info = env.step(action, return_step_reward=True)
            
            # 4. 获取奖励
            reward = step_info['step_reward'][0]
            done = step_info['done']
            
            episode_reward += reward
            
            # 5. [这里Graph-R1会存储经验并更新策略]
            # experience = (state, action, reward, next_state, done)
            # replay_buffer.add(experience)
            # policy.update(replay_buffer.sample(batch_size))
            
            # 6. 检查是否完成
            if done or step >= max_steps - 1:
                print(f"  Episode结束: 步数={step+1}, "
                      f"最终掌握度={step_info['current_target_score'][0]:.4f}, "
                      f"总奖励={episode_reward:.4f}")
                break
    
    print("\n✅ Graph-R1接口演示完成")


if __name__ == '__main__':
    demo_kg_sampling()
    demo_graph_r1_interface()

