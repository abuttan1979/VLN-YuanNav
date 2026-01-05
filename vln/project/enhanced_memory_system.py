"""
增强记忆系统 - 多层次记忆架构
解决agent缺乏记忆力和空间认知能力的问题
"""
import numpy as np
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import time

class MemoryType(Enum):
    SPATIAL = "spatial"        # 空间记忆（位置、方向、区域）
    SEMANTIC = "semantic"      # 语义记忆（物体、关系、目标）
    EPISODIC = "episodic"      # 情景记忆（时间序列经历）
    PROCEDURAL = "procedural"  # 程序记忆（策略、模式）

@dataclass
class SpatialMemory:
    """空间记忆单元"""
    position: Tuple[float, float, float]  # (x, y, z)
    rotation: Tuple[float, float, float, float]  # quaternion
    region_id: str  # 房间/区域标识
    explored_level: float  # 探索程度 [0, 1]
    landmarks: List[str]  # 地标物体
    connections: List[str]  # 连接区域
    
@dataclass
class SemanticMemory:
    """语义记忆单元"""
    object_type: str  # 物体类型
    object_id: str   # 物体唯一标识
    location: Optional[Tuple[float, float, float]]  # 位置
    properties: Dict[str, Any]  # 属性
    relationships: Dict[str, str]  # 与其他物体的关系
    confidence: float  # 确信度 [0, 1]
    last_seen_step: int  # 最后看到时的步数

@dataclass
class EpisodicMemory:
    """情景记忆单元"""
    step: int
    action: str
    perception: Dict
    result: str  # 成功/失败/部分成功
    importance: float  # 重要性评分 [0, 1]
    timestamp: float
    context: Dict[str, Any]  # 上下文信息

class EnhancedMemorySystem:
    """
    多层次记忆系统 - 解决agent记忆力问题
    
    包含四个层次的记忆：
    1. 工作记忆：当前推理上下文
    2. 情景记忆：按时间序列的经历
    3. 语义记忆：物体关系和空间结构
    4. 程序记忆：导航策略和模式
    """
    
    def __init__(self, max_episodic=200, max_semantic=1000, max_spatial=500):
        # 工作记忆 - 当前推理上下文
        self.working_memory = {
            'current_goal': None,
            'current_focus': None,
            'recent_actions': deque(maxlen=10),
            'hypothesis': [],
            'reasoning_chain': []
        }
        
        # 情景记忆 - 时间序列经历
        self.episodic_memory = deque(maxlen=max_episodic)
        
        # 语义记忆 - 物体和关系
        self.semantic_memory = {}  # {object_id: SemanticMemory}
        self.object_locations = defaultdict(list)  # {object_type: [location1, location2, ...]}
        
        # 空间记忆 - 环境地图
        self.spatial_memory = {}  # {region_id: SpatialMemory}
        self.environment_graph = defaultdict(list)  # 区域连接图
        self.current_region = "unknown"
        
        # 程序记忆 - 导航策略
        self.strategies = {
            'exploration': [],
            'target_search': [],
            'obstacle_avoidance': [],
            'recovery': []
        }
        
        # 记忆统计
        self.stats = {
            'total_episodes': 0,
            'total_objects': 0,
            'total_regions': 0,
            'navigation_successes': 0,
            'navigation_failures': 0
        }
        
    def store_episodic(self, step: int, action: str, perception: Dict, 
                      result: str = "partial", importance: float = 0.5):
        """存储情景记忆"""
        episode = EpisodicMemory(
            step=step,
            action=action,
            perception=perception,
            result=result,
            importance=importance,
            timestamp=time.time(),
            context={
                'region': self.current_region,
                'objects_visible': len(perception.get('objects', [])),
                'navigation_success': result in ['success', 'partial']
            }
        )
        
        self.episodic_memory.append(episode)
        self.stats['total_episodes'] += 1
        
        # 更新程序记忆
        self._update_procedural_memory(action, result, importance)
        
    def _associate_object(self, object_type: str, location: Optional[Tuple[float, float, float]], 
                         threshold: float = 2.0) -> Optional[str]:
        """
        基于空间距离的物体关联
        
        Args:
            object_type: 物体类型
            location: 物体位置 (x, y, z)
            threshold: 距离阈值（米），默认2米
        
        Returns:
            如果找到匹配的物体，返回其 object_id；否则返回 None
        """
        if location is None:
            return None
        
        current_pos = np.array(location)
        best_match_id = None
        min_distance = float('inf')
        
        for obj_id, semantic in self.semantic_memory.items():
            if semantic.object_type == object_type and semantic.location:
                stored_pos = np.array(semantic.location)
                distance = np.linalg.norm(current_pos - stored_pos)
                
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    best_match_id = obj_id
        
        if best_match_id is not None:
            print(f"🔗 [物体关联] 识别到已知物体: {object_type} (距离: {min_distance:.2f}m)")
        
        return best_match_id
    
    def store_semantic(self, object_type: str, object_id: str, location: Optional[Tuple[float, float, float]], 
                      properties: Dict[str, Any], confidence: float = 0.8):
        """存储语义记忆 - 支持物体重识别
        
        Returns:
            Dict: 包含关联信息 {'is_new': bool, 'object_id': str, 'distance': float or None}
        """
        
        # 检查是否是已知的物体
        existing_object_id = self._associate_object(object_type, location)
        
        association_info = {
            'is_new': existing_object_id is None,
            'object_id': existing_object_id if existing_object_id else object_id,
            'distance': None
        }
        
        if existing_object_id is not None:
            # 更新已存在的物体记忆
            existing_semantic = self.semantic_memory[existing_object_id]
            
            # 计算距离
            if location and existing_semantic.location:
                old_pos = np.array(existing_semantic.location)
                new_pos = np.array(location)
                distance = np.linalg.norm(new_pos - old_pos)
                association_info['distance'] = distance
            
            # 更新位置（使用加权平均，新观测权重更高）
            if location and existing_semantic.location:
                old_pos = np.array(existing_semantic.location)
                new_pos = np.array(location)
                alpha = 0.3  # 新观测的权重
                updated_pos = (1 - alpha) * old_pos + alpha * new_pos
                existing_semantic.location = tuple(updated_pos)
            elif location:
                existing_semantic.location = location
            
            # 更新置信度（使用最大值）
            existing_semantic.confidence = max(existing_semantic.confidence, confidence)
            
            # 更新最后看到的步数
            existing_semantic.last_seen_step = len(self.episodic_memory)
            
            # 合并属性
            existing_semantic.properties.update(properties)
            
            print(f"🔄 [语义记忆] 更新物体: {object_type} (ID: {existing_object_id})")
            return association_info
        
        # 如果是新物体，创建新的语义记忆
        semantic = SemanticMemory(
            object_type=object_type,
            object_id=object_id,
            location=location,
            properties=properties,
            relationships={},
            confidence=confidence,
            last_seen_step=len(self.episodic_memory)
        )
        
        self.semantic_memory[object_id] = semantic
        
        if location:
            self.object_locations[object_type].append(location)
            
        self.stats['total_objects'] += 1
        print(f"➕ [语义记忆] 新增物体: {object_type} (ID: {object_id})")
        return association_info
        
    def store_spatial(self, position: Tuple[float, float, float], rotation: Tuple[float, float, float, float],
                     region_id: str = None, landmarks: List[str] = None):
        """存储空间记忆"""
        if region_id is None:
            region_id = self._classify_region(position)
            
        if landmarks is None:
            landmarks = self._extract_landmarks(position)
            
        if region_id not in self.spatial_memory:
            self.spatial_memory[region_id] = SpatialMemory(
                position=position,
                rotation=rotation,
                region_id=region_id,
                explored_level=0.0,
                landmarks=landmarks,
                connections=[]
            )
            self.stats['total_regions'] += 1
            
        self.current_region = region_id
        
        # 更新区域连接
        self._update_spatial_connections(region_id)
        
    def retrieve_relevant(self, query_type: str, query: Any, k: int = 5) -> List[Any]:
        """检索相关记忆"""
        if query_type == "episodic":
            return self._retrieve_episodic(query, k)
        elif query_type == "semantic":
            return self._retrieve_semantic(query, k)
        elif query_type == "spatial":
            return self._retrieve_spatial(query, k)
        elif query_type == "procedural":
            return self._retrieve_procedural(query, k)
        else:
            return []
            
    def _retrieve_episodic(self, query: str, k: int) -> List[EpisodicMemory]:
        """检索情景记忆"""
        relevant = []
        query_lower = query.lower()
        
        # 按重要性和时间排序
        episodes = list(self.episodic_memory)
        episodes.sort(key=lambda x: (x.importance, x.step), reverse=True)
        
        for episode in episodes:
            if (query_lower in episode.action.lower() or 
                query_lower in episode.result.lower() or
                any(query_lower in str(v).lower() for v in episode.context.values())):
                relevant.append(episode)
                if len(relevant) >= k:
                    break
                    
        return relevant
        
    def _retrieve_semantic(self, query: str, k: int) -> List[SemanticMemory]:
        """检索语义记忆"""
        relevant = []
        query_lower = query.lower()
        
        for semantic in self.semantic_memory.values():
            if (query_lower in semantic.object_type.lower() or
                query_lower in semantic.object_id.lower() or
                any(query_lower in str(v).lower() for v in semantic.properties.values())):
                relevant.append(semantic)
                if len(relevant) >= k:
                    break
                    
        return relevant
        
    def _retrieve_spatial(self, query: Any, k: int) -> List[SpatialMemory]:
        """检索空间记忆"""
        if isinstance(query, str):
            # 按区域检索
            return [self.spatial_memory[region] for region in [query] if region in self.spatial_memory][:k]
        elif isinstance(query, tuple) and len(query) == 3:
            # 按位置检索附近区域
            query_pos = np.array(query)
            regions = []
            
            for region in self.spatial_memory.values():
                region_pos = np.array(region.position)
                distance = np.linalg.norm(query_pos - region_pos)
                if distance < 5.0:  # 5米范围内
                    regions.append((region, distance))
                    
            regions.sort(key=lambda x: x[1])
            return [region for region, _ in regions][:k]
            
        return []
        
    def _retrieve_procedural(self, query: str, k: int) -> List[str]:
        """检索程序记忆"""
        if query in self.strategies:
            return self.strategies[query][-k:]
        return []
        
    def _update_procedural_memory(self, action: str, result: str, importance: float):
        """更新程序记忆"""
        if importance > 0.7:  # 高重要性经验
            if result == "success":
                strategy_type = self._classify_strategy_type(action)
                if strategy_type in self.strategies:
                    self.strategies[strategy_type].append(action)
                    
    def _classify_strategy_type(self, action: str) -> str:
        """分类策略类型"""
        if "forward" in action:
            return "exploration"
        elif "turn" in action:
            return "target_search"
        elif "backward" in action:
            return "recovery"
        else:
            return "obstacle_avoidance"
            
    def _classify_region(self, position: Tuple[float, float, float]) -> str:
        """根据位置分类区域"""
        x, y, z = position
        
        # 简单的区域分类逻辑
        if y > 1.5:  # 高度判断
            return f"upper_level_{x//5}_{z//5}"
        else:
            return f"room_{x//5}_{z//5}"
            
    def _extract_landmarks(self, position: Tuple[float, float, float]) -> List[str]:
        """提取地标"""
        landmarks = []
        x, y, z = position
        
        # 根据位置特征识别地标
        if abs(x) < 2 and abs(z) < 2:
            landmarks.append("center_area")
        if abs(x) > 8 or abs(z) > 8:
            landmarks.append("boundary_area")
            
        return landmarks
        
    def _update_spatial_connections(self, region_id: str):
        """更新空间连接"""
        # 简化的连接更新逻辑
        if region_id not in self.environment_graph:
            self.environment_graph[region_id] = []
            
    def get_spatial_context(self, current_pos: Tuple[float, float, float]) -> Dict[str, Any]:
        """获取空间上下文"""
        current_region = self._classify_region(current_pos)
        nearby_regions = self._retrieve_spatial(current_pos, k=5)
        
        return {
            'current_region': current_region,
            'explored_regions': len(self.spatial_memory),
            'exploration_progress': np.mean([r.explored_level for r in self.spatial_memory.values()]),
            'nearby_landmarks': nearby_regions,
            'navigation_options': self.environment_graph.get(current_region, [])
        }
        
    def get_semantic_context(self, object_type: str = None) -> Dict[str, Any]:
        """获取语义上下文"""
        if object_type:
            locations = self.object_locations.get(object_type, [])
            objects = [obj for obj in self.semantic_memory.values() if obj.object_type == object_type]
        else:
            locations = []
            objects = list(self.semantic_memory.values())
            
        return {
            'known_objects': len(objects),
            'object_locations': locations,
            'object_types': list(self.object_locations.keys()),
            'high_confidence_objects': [obj for obj in objects if obj.confidence > 0.8]
        }
        
    def get_navigation_guidance(self, target_type: str = "lamp") -> Dict[str, Any]:
        """获取导航指导"""
        # 查找目标物体的历史位置
        target_objects = [obj for obj in self.semantic_memory.values() 
                         if target_type.lower() in obj.object_type.lower()]
        
        if target_objects:
            # 按置信度和最近性排序
            target_objects.sort(key=lambda x: (x.confidence, x.last_seen_step), reverse=True)
            best_target = target_objects[0]
            
            guidance = {
                'target_found': True,
                'estimated_location': best_target.location,
                'confidence': best_target.confidence,
                'search_strategy': 'direct_approach' if best_target.confidence > 0.8 else 'systematic_search'
            }
        else:
            # 没有找到目标，建议搜索策略
            guidance = {
                'target_found': False,
                'search_strategy': 'exploration',
                'exploration_suggestions': self.strategies.get('exploration', [])
            }
            
        return guidance
        
    def consolidate_memory(self):
        """记忆巩固"""
        # 巩固重要记忆
        important_episodes = [ep for ep in self.episodic_memory if ep.importance > 0.8]
        
        # 更新空间探索程度
        for region in self.spatial_memory.values():
            if region.region_id in [ep.context.get('region') for ep in important_episodes]:
                region.explored_level = min(1.0, region.explored_level + 0.1)
                
        print(f"🧠 [记忆巩固] 已巩固 {len(important_episodes)} 条重要记忆")
        
    def export_memory(self, file_path: str):
        """导出记忆数据"""
        # 处理无法直接序列化的对象
        def convert_deque_to_list(obj):
            if isinstance(obj, deque):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_deque_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_deque_to_list(item) for item in obj]
            else:
                return obj
        
        memory_data = {
            'working_memory': convert_deque_to_list(dict(self.working_memory)),
            'episodic_memory': [asdict(ep) for ep in self.episodic_memory],
            'semantic_memory': {k: asdict(v) for k, v in self.semantic_memory.items()},
            'spatial_memory': {k: asdict(v) for k, v in self.spatial_memory.items()},
            'stats': dict(self.stats),
            'environment_graph': dict(self.environment_graph),
            'object_locations': dict(self.object_locations),
            'export_timestamp': time.time()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2, default=str)
            
        print(f"💾 [记忆导出] 记忆数据已保存到 {file_path}")
        
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        return {
            'episodic_count': len(self.episodic_memory),
            'semantic_count': len(self.semantic_memory),
            'spatial_count': len(self.spatial_memory),
            'current_region': self.current_region,
            'navigation_success_rate': (
                self.stats['navigation_successes'] / 
                max(1, self.stats['navigation_successes'] + self.stats['navigation_failures'])
            ),
            'exploration_progress': len(self.spatial_memory) / 50.0,  # 假设目标探索50个区域
            'working_memory_load': len(self.working_memory['recent_actions'])
        }