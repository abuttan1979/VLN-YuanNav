"""
VLN (Vision-Language Navigation) Agent - 记忆增强版
包含十个核心模块的完整实现 + 多层次记忆系统
"""
import habitat_sim
import cv2
import numpy as np
import torch
import json
import os
from PIL import Image
import random
from typing import Dict, List, Tuple, Optional, Any
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from dataclasses import dataclass
from collections import deque
import habitat_sim
import matplotlib.pyplot as plt
import datetime
from scipy.spatial.transform import Rotation as R # 用于处理朝向

# 导入增强记忆系统
import sys
sys.path.append('/home/ubuntu/YuanNav/vln/project')
from enhanced_memory_system import EnhancedMemorySystem, MemoryType

def draw_trajectory(agent, target_pos=None, save_path="vln_path_pro.png"):
    """绘制专业版轨迹图：包含墙壁边界、1:1比例和朝向"""
    trajectory = agent.memory_mod.get_trajectory()
    if not trajectory:
        print("❌ [绘图失败] 轨迹数据为空")
        return

    # 1. 提取路径点 (X, Z)
    path_points = np.array([(e['perception']['spatial']['position'][0], 
                             e['perception']['spatial']['position'][2]) for e in trajectory])

    # 2. 提取朝向向量
    def get_heading_vector(rotation_quat):
        quat_list = [rotation_quat.x, rotation_quat.y, rotation_quat.z, rotation_quat.w]
        r = R.from_quat(quat_list)
        # Habitat 偏航角在 Y 轴，使用 YXZ 序列修复之前的报错
        yaw = r.as_euler('YXZ')[0] 
        # 返回 (dx, dz) 向量
        return np.array([np.sin(yaw), np.cos(yaw)])

    start_dir = get_heading_vector(trajectory[0]['perception']['spatial']['rotation'])
    end_dir = get_heading_vector(trajectory[-1]['perception']['spatial']['rotation'])

    # 3. 绘图设置
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- [关键增强] 绘制真实的地图轮廓 ---
    # 采样 5000 个可导航点，用极小的点和低透明度描绘出“地板”形状
    map_points = np.array([agent.pathfinder.get_random_navigable_point() for _ in range(5000)])
    ax.scatter(map_points[:, 0], map_points[:, 2], color='lightgray', s=1, alpha=0.3, label='Navigable Area')

    # 绘制行走轨迹
    ax.plot(path_points[:, 0], path_points[:, 1], color='#1f77b4', linewidth=2.5, zorder=3, label='Agent Path')

    # 绘制起点（绿）和终点（红）
    ax.scatter(path_points[0, 0], path_points[0, 1], color='green', s=100, marker='o', zorder=5)
    ax.scatter(path_points[-1, 0], path_points[-1, 1], color='red', s=100, marker='X', zorder=5)

    # 绘制朝向箭头 (Quiver)
    # scale 越小箭头越长
    ax.quiver(path_points[0, 0], path_points[0, 1], start_dir[0], start_dir[1], 
              color='green', scale=15, width=0.008, headwidth=5, zorder=6, label='Start Facing')
    ax.quiver(path_points[-1, 0], path_points[-1, 1], end_dir[0], end_dir[1], 
              color='red', scale=15, width=0.008, headwidth=5, zorder=6, label='End Facing')

    # 4. 绘制目标 (LAMP)
    if target_pos is not None:
        ax.scatter(target_pos[0], target_pos[2], color='#ff7f0e', s=250, marker='*', 
                   edgecolors='black', linewidths=1, zorder=7, label='Target (LAMP)')
        # 连线显示误差
        ax.plot([path_points[-1, 0], target_pos[0]], [path_points[-1, 1], target_pos[2]], 
                ':', color='gray', alpha=0.6)

    # --- [精细化坐标系] ---
    ax.set_aspect('equal', adjustable='box') # 强制 1:1 比例
    ax.set_title(f"Visual Navigation Trajectory Analysis\nScene: Apartment | Steps: {len(path_points)}", fontsize=12)
    ax.set_xlabel("World X (meters)", fontsize=10)
    ax.set_ylabel("World Z (meters)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 图例放在外面防止遮挡地图
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📍 [系统] 优化版轨迹图已保存: {save_path}")

# ============================================================================
# 模块1: 指令 (Instruction)agent.run(instruction, target_pos=target_pos)
# ============================================================================
class InstructionModule:
    """指令模块：接收和处理导航指令"""
    
    def __init__(self):
        self.current_instruction = None
        self.instruction_history = []
    
    def set_instruction(self, instruction: str):
        """设置当前导航指令"""
        self.current_instruction = instruction
        self.instruction_history.append(instruction)
        print(f"📝 [指令模块] 收到指令: {instruction}")
        return instruction
    
    def get_instruction(self) -> Optional[str]:
        """获取当前指令"""
        return self.current_instruction
    
    def parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """解析指令，提取关键信息"""
        return {
            "raw": instruction,
            "goal": instruction,
            "type": "navigation"
        }


# ============================================================================
# 模块2: 规划 (Planning)execute
# ============================================================================
class PlanningModule:
    """规划模块：基于指令生成导航计划"""
    
    def __init__(self, llm_model=None, llm_processor=None):
        self.llm_model = llm_model
        self.llm_processor = llm_processor
        self.plan_history = []
    
    def generate_plan(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """基于指令生成导航计划"""
        print(f"🧠 [规划模块] 正在为指令生成计划: {instruction['goal']}")
        
        # 简单规则规划 (可扩展为LLM规划)
        plan = {
            "steps": ["explore", "navigate", "search"],
            "strategy": "systematic_search",
            "max_steps": 100
        }
        
        self.plan_history.append(plan)
        print(f"✅ [规划模块] 计划生成完成: {plan}")
        return plan


# ============================================================================
# 模块3: 计划 (Plan)
# ============================================================================
@dataclass
class Plan:
    """计划数据结构"""
    steps: List[str]
    strategy: str
    max_steps: int
    current_step: int = 0
    
    def get_current_step(self) -> str:
        """获取当前步骤"""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return "complete"
    
    def advance_step(self):
        """推进到下一步"""
        self.current_step += 1


# ============================================================================
# 模块4: 感知 + 空间表达 (深度增强版)
# ============================================================================
class PerceptionModule:
    """感知模块：处理 RGB 视觉（CLAHE增强）、深度信息（避障）和空间状态"""
    
    def __init__(self, simulator=None):
        self.simulator = simulator
        self.observation_history = []
    
    def get_safe_distance(self, depth_obs: np.ndarray) -> float:
        """
        计算视野中心区域的最小深度值
        :param depth_obs: Habitat 提供的深度观测值 (H, W)
        :return: 前方障碍物的最近距离（米）
        """
        if depth_obs is None:
            return 999.0
            
        # 1. 定义中心感兴趣区域 (ROI: Region of Interest)
        # 我们关注屏幕中心 1/3 的区域，避免边缘干扰
        h, w = depth_obs.shape
        h_start, h_end = h // 3, 2 * h // 3
        w_start, w_end = w // 3, 2 * w // 3
        center_zone = depth_obs[h_start:h_end, w_start:w_end]
        
        # 2. 过滤掉无效深度（Habitat中0通常代表超远或无效，需根据配置确定）
        # 大部分情况下直接取最小值即可代表最近障碍物
        min_dist = np.min(center_zone)
        
        return float(min_dist)
    
    def perceive(self, agent_state=None) -> Dict[str, Any]:
        """执行全模态感知"""
        if self.simulator is None:
            return {"error": "simulator not initialized"}
        
        # 获取所有传感器观测
        obs = self.simulator.get_sensor_observations()
        rgb_image = obs.get("color_sensor", None)
        depth_map = obs.get("depth_sensor", None)
        semantic_map = obs.get("semantic_sensor", None)
        
        # --- 1. RGB 图像处理 (CLAHE 增强) ---
        image_pil = None
        rgb_bgr_output = None

        if rgb_image is not None:
            # 基础转换 (RGBA -> BGR)
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)
            
            # CLAHE 对比度受限的自适应直方图均衡化
            lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            rgb_bgr_output = enhanced_bgr
            rgb_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(rgb_rgb)
        
        # --- 2. 深度信息处理 ---
        dist_to_obstacle = 999.0
        if depth_map is not None:
            dist_to_obstacle = self.get_safe_distance(depth_map)
        
        # --- 3. 语义信息处理 (物体检测) ---
        detected_objects = []
        if semantic_map is not None:
            detected_objects = self._detect_objects_from_semantic(semantic_map)
        
        # --- 4. 空间状态提取 ---
        spatial_info = self._extract_spatial_info(agent_state)
        
        # 封装感知结果
        perception = {
            "image": image_pil,           # 用于 LLM 思考的增强图像
            "image_array": rgb_bgr_output, # 原始数组供 OpenCV 使用
            "depth": dist_to_obstacle,    # 前方障碍物距离 (米)
            "objects": detected_objects,  # 检测到的物体列表
            "spatial": spatial_info,       # 坐标与朝向
            "timestamp": len(self.observation_history)
        }
        
        self.memory_optimize(perception) # 可选：记录历史
        self.observation_history.append(perception)
        return perception
    
    def _detect_objects_from_semantic(self, semantic_map: np.ndarray) -> List[Dict[str, Any]]:
        """从语义图中检测物体"""
        objects = []
        
        if semantic_map is None:
            return objects
        
        # 获取场景中的所有物体标签
        try:
            scene = self.simulator.get_active_scene()
            object_ids = np.unique(semantic_map)
            
            # 过滤掉背景标签（通常是0）
            object_ids = object_ids[object_ids > 0]
            
            for obj_id in object_ids:
                # 计算该物体在图像中的占比
                mask = (semantic_map == obj_id)
                area = np.sum(mask)
                total_pixels = semantic_map.size
                area_ratio = area / total_pixels
                
                # 只记录占比超过1%的物体
                if area_ratio > 0.01:
                    # 尝试获取物体名称
                    obj_name = f"object_{obj_id}"
                    try:
                        obj = scene.get_object_by_id(obj_id)
                        if hasattr(obj, 'category_name'):
                            obj_name = obj.category_name
                        elif hasattr(obj, 'semantic_id'):
                            obj_name = f"semantic_{obj.semantic_id}"
                    except:
                        pass
                    
                    objects.append({
                        "type": obj_name,
                        "id": int(obj_id),
                        "area_ratio": float(area_ratio),
                        "confidence": min(1.0, area_ratio * 10)  # 根据面积估算置信度
                    })
        except Exception as e:
            # 如果语义解析失败，返回空列表
            pass
        
        return objects

    def _extract_spatial_info(self, agent_state) -> Dict[str, Any]:
        """提取 Agent 当前位置和旋转"""
        if agent_state is None:
            return {"position": None, "rotation": None, "has_position": False}
        return {
            "position": agent_state.position.tolist(),
            "rotation": agent_state.rotation,
            "has_position": True
        }

    def memory_optimize(self, perception: Dict):
        """简单的日志记录，防止内存中存储过大的观测历史"""
        if len(self.observation_history) % 20 == 0 and len(self.observation_history) > 0:
            print(f"👁️ [感知系统] 步数: {perception['timestamp']}, 当前前方深度: {perception['depth']:.2f}m")


# ============================================================================
# 模块5: 记忆 (Memory) - 增强版
# ============================================================================
class MemoryModule:
    """记忆模块：集成增强记忆系统，支持多层次记忆"""
    
    def __init__(self, max_memory_size=100):
        # 使用增强记忆系统替代简单的队列
        self.enhanced_memory = EnhancedMemorySystem(
            max_episodic=200, 
            max_semantic=1000, 
            max_spatial=500
        )
        
        # 保持向后兼容的接口
        self.memory = deque(maxlen=max_memory_size)
        self.last_result = "partial"  # 用于重要性评估
        
    def store(self, perception: Dict, action: str, reward: float = 0.0) -> Dict[str, Any]:
        """存储一步的经验 - 同时存储到简单记忆和增强记忆系统
        
        Returns:
            Dict: 包含物体关联信息 {'associations': List[Dict]}
        """
        step = len(self.memory)
        associations = []  # 存储物体关联信息
        
        # 存储到简单记忆（保持向后兼容）
        memory_entry = {
            "perception": perception,
            "action": action,
            "step": step
        }
        self.memory.append(memory_entry)
        
        # 存储到增强记忆系统
        if perception.get('objects'):
            for obj in perception['objects']:
                if obj.get('type') and obj.get('confidence', 0) > 0.5:
                    # 存储语义记忆
                    association_info = self.enhanced_memory.store_semantic(
                        object_type=obj['type'],
                        object_id=f"{obj['type']}_{step}_{random.randint(1000, 9999)}",
                        location=tuple(perception.get('position', [0, 0, 0])) if perception.get('position') else None,
                        properties={'confidence': obj.get('confidence', 0), 'bbox': obj.get('bbox', [])},
                        confidence=obj.get('confidence', 0)
                    )
                    # 记录关联信息
                    associations.append({
                        'object_type': obj['type'],
                        'is_new': association_info['is_new'],
                        'object_id': association_info['object_id'],
                        'distance': association_info['distance']
                    })
        
        # 存储空间记忆
        if perception.get('position') and perception.get('rotation'):
            self.enhanced_memory.store_spatial(
                position=tuple(perception['position']),
                rotation=tuple(perception['rotation']),
                region_id=None,  # 让系统自动分类
                landmarks=[obj['type'] for obj in perception.get('objects', []) if obj.get('confidence', 0) > 0.7]
            )
        
        # 存储情景记忆
        importance = self._calculate_importance(action, self.last_result, step)
        self.enhanced_memory.store_episodic(
            step=step,
            action=action,
            perception=perception,
            result=self.last_result,
            importance=importance
        )
        
        print(f"💾 [增强记忆模块] 存储第 {step} 步经验 (重要性: {importance:.2f})")
        
        return {'associations': associations}
    
    def retrieve(self, query: str = None, k: int = 5) -> List[Dict]:
        """检索记忆 - 增强版检索能力"""
        if query is None:
            # 默认检索最近的记忆
            return list(self.memory)[-k:]
        
        # 使用增强记忆系统进行智能检索
        episodic_results = self.enhanced_memory.retrieve_relevant("episodic", query, k=k//2)
        semantic_results = self.enhanced_memory.retrieve_relevant("semantic", query, k=k//2)
        
        # 转换格式以保持兼容性
        combined_results = []
        
        # 添加情景记忆
        for episode in episodic_results:
            combined_results.append({
                "perception": episode.perception,
                "action": episode.action,
                "step": episode.step,
                "importance": episode.importance,
                "result": episode.result
            })
        
        # 添加语义记忆
        for semantic in semantic_results:
            combined_results.append({
                "object_type": semantic.object_type,
                "location": semantic.location,
                "confidence": semantic.confidence,
                "last_seen_step": semantic.last_seen_step
            })
        
        # 如果没有找到相关记忆，返回最近的记忆
        if not combined_results:
            return list(self.memory)[-k:]
            
        return combined_results[:k]
    
    def get_trajectory(self) -> List[Dict]:
        """获取轨迹 - 兼容原接口"""
        return list(self.memory)
    
    def get_spatial_context(self, current_position: Tuple[float, float, float]) -> Dict[str, Any]:
        """获取空间上下文信息"""
        return self.enhanced_memory.get_spatial_context(current_position)
    
    def get_semantic_context(self, object_type: str = None) -> Dict[str, Any]:
        """获取语义上下文信息"""
        return self.enhanced_memory.get_semantic_context(object_type)
    
    def get_navigation_guidance(self, target_type: str = "lamp") -> Dict[str, Any]:
        """获取导航指导"""
        return self.enhanced_memory.get_navigation_guidance(target_type)
    
    def set_last_result(self, result: str):
        """设置上一步的结果，用于重要性评估"""
        self.last_result = result
    
    def _calculate_importance(self, action: str, result: str, step: int) -> float:
        """计算记忆重要性"""
        importance = 0.5  # 基础重要性
        
        # 根据结果调整重要性
        if result == "success":
            importance += 0.3
        elif result == "failure":
            importance += 0.2
        
        # 根据动作调整重要性
        if "stop" in action:
            importance += 0.1
        elif "turn" in action:
            importance += 0.05
        
        # 根据步数调整（较新的记忆更重要）
        importance += min(0.2, step * 0.001)
        
        return min(1.0, importance)
    
    def consolidate_memory(self):
        """记忆巩固"""
        self.enhanced_memory.consolidate_memory()
    
    def export_memory(self, file_path: str):
        """导出记忆数据"""
        self.enhanced_memory.export_memory(file_path)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        summary = self.enhanced_memory.get_memory_summary()
        summary.update({
            'simple_memory_size': len(self.memory),
            'backward_compatible': True
        })
        return summary


# ============================================================================
# 模块6: 跨模态对齐 (修复 Action 解析 Bug)ExecutionModule
# ============================================================================
class CrossModalAlignmentModule:
    def __init__(self, model=None, processor=None, memory_module=None):
        self.model = model
        self.processor = processor
        self.memory_module = memory_module  # 集成记忆模块

    def think(self, perception: Dict, instruction: str, memory: List[Dict] = None, 
              collision_warning: bool = False, step_count: int = 0) -> Dict[str, Any]:
        if self.model is None: return {"action": "move_forward", "reasoning": "default"}
        
        image = perception.get("image")
        if image is None: return {"action": "stop", "reasoning": "blind"}
        
        # === [修复] 直接使用外部传入的 step_count，不被 memory 覆盖 ===
        current_step = step_count 
        
        # === [记忆增强] 获取上下文信息 ===
        spatial_context = {}
        semantic_context = {}
        navigation_guidance = {}
        
        if self.memory_module:
            # 获取空间上下文
            if perception.get('position'):
                spatial_context = self.memory_module.get_spatial_context(perception['position'])
            
            # 获取语义上下文（专注台灯相关）
            semantic_context = self.memory_module.get_semantic_context("lamp")
            
            # 获取导航指导
            navigation_guidance = self.memory_module.get_navigation_guidance("lamp")
        
        status_prompt = ""
        if collision_warning:
            status_prompt = "\n[⚠️ WARNING: STUCK! Last move hit a wall. TURN to find open space!]"
        
        # === [记忆增强] 构建记忆感知的提示词 ===
        memory_context = self._build_memory_context(spatial_context, semantic_context, navigation_guidance)
        
        text_prompt = f"""Task: Find the LAMP and STOP when you see it clearly.

Current Step: {current_step} (Max: 300) {status_prompt}

{memory_context}

IMPORTANT INSTRUCTIONS:
1. If you see a lamp clearly in your view, you MUST output "Action: stop" immediately.
2. DO NOT keep turning if you see the lamp 

First, describe what you see and your reasoning. Then, output your action.
Format:
Reasoning: [your reasoning here]
Action: [move_forward / turn_left / turn_right / stop / move_backward]"""

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pure_output = output_text.replace(text, "").strip() if text in output_text else output_text
        
        # 只提取模型的推理部分（"assistant" 之后的内容）
        if "assistant" in pure_output:
            pure_output = pure_output.split("assistant")[-1].strip()
        
        # 提取推理部分（"Reasoning:" 和 "Action:" 之间的内容）
        reasoning = pure_output
        action = "move_forward"  # 默认动作
        
        if "Reasoning:" in pure_output and "Action:" in pure_output:
            parts = pure_output.split("Action:")
            reasoning = parts[0].replace("Reasoning:", "").strip()
            action_part = parts[1].strip().split()[0] if parts[1].strip() else "move_forward"
            action = action_part.lower()
        elif "Action:" in pure_output:
            # 如果只有Action，提取动作
            parts = pure_output.split("Action:")
            reasoning = parts[0].strip()
            action_part = parts[1].strip().split()[0] if parts[1].strip() else "move_forward"
            action = action_part.lower()
        else:
            # 如果没有格式，直接使用整个输出
            reasoning = pure_output
            action = self._extract_action(pure_output)
        
        print(f"🧠 [CoT推理] Step {current_step}: {reasoning}")
        print(f"🎯 [动作选择] Step {current_step}: {action}")
        
        return {"action": action, "reasoning": reasoning}
    
    def _build_memory_context(self, spatial_context: Dict, semantic_context: Dict, navigation_guidance: Dict) -> str:
        """构建记忆感知的上下文信息"""
        context_parts = []
        
        # 空间上下文
        if spatial_context:
            explored_regions = spatial_context.get('explored_regions', 0)
            exploration_progress = spatial_context.get('exploration_progress', 0)
            context_parts.append(f"[SPATIAL MEMORY] Explored {explored_regions} regions, {exploration_progress:.1%} completion")
            
            # 添加附近地标信息
            nearby_landmarks = spatial_context.get('nearby_landmarks', [])
            if nearby_landmarks:
                landmark_names = [landmark.region_id for landmark in nearby_landmarks[:3]]
                context_parts.append(f"Nearby landmarks: {', '.join(landmark_names)}")
        
        # 语义上下文
        if semantic_context:
            known_objects = semantic_context.get('known_objects', 0)
            if known_objects > 0:
                context_parts.append(f"[SEMANTIC MEMORY] Previously seen {known_objects} objects")
                
            # 添加高置信度物体
            high_confidence = semantic_context.get('high_confidence_objects', [])
            if high_confidence:
                object_types = [obj.object_type for obj in high_confidence[:3]]
                context_parts.append(f"High-confidence objects: {', '.join(object_types)}")
        
        # 导航指导
        if navigation_guidance:
            target_found = navigation_guidance.get('target_found', False)
            if target_found:
                confidence = navigation_guidance.get('confidence', 0)
                context_parts.append(f"[NAVIGATION GUIDANCE] Target likely found with {confidence:.1%} confidence")
            else:
                search_strategy = navigation_guidance.get('search_strategy', 'exploration')
                context_parts.append(f"[NAVIGATION GUIDANCE] Search strategy: {search_strategy}")
        
        if context_parts:
            return "\n".join(context_parts) + "\n"
        else:
            return "[MEMORY] Starting fresh exploration\n"

    def _extract_action(self, text: str) -> str:
        text_lower = text.lower()
        # 提取 Action 部分
        if "action:" in text_lower:
            action_part = text_lower.split("action:")[-1].strip()
        else:
            lines = text_lower.strip().split('\n')
            action_part = lines[-1] if lines else ""

        # --- [策略干预] 门框优先级与推进动力 ---
        # 减少门框的强制干预，让模型有更多自主决策权
        is_exploring_door = any(word in text_lower for word in ["doorway", "entrance", "enter", "another room", "opening"])
        
        # 只在没有明确目标（台灯）时才应用门框引导
        has_lamp = any(word in text_lower for word in ["lamp", "light", "shade"])
        
        if is_exploring_door and "stop" not in action_part and not has_lamp:
            if "forward" not in action_part:
                print("✨ [语义引导] 检测到门框且无目标，修正动作：move_forward 以增强探索。")
                return "move_forward"

        # --- [审查机制] 目标物校验 (由沙发改为台灯) ---
        # 检查推理中是否明确提到了 lamp
        has_lamp_mentioned = any(word in text_lower for word in ["lamp", "light", "shade"])
        is_visible = any(word in text_lower for word in ["visible", "see", "found", "center", "front", "close", "near", "looking", "view"])
        is_located = any(word in text_lower for word in ["located", "near", "behind", "next to", "beside"])
        
        # 如果推理中明确提到看到了 lamp，但 action 不是 stop，则强制修正为 stop
        if has_lamp_mentioned and (is_visible or is_located) and "stop" not in action_part:
            print(f"🎯 [强制修正] 推理中提到 lamp，但 action 不是 stop，强制修正为 stop！")
            return "stop"
        
        if "stop" in action_part:
            # 进一步放宽：只要推理中提到目标物，就允许停止
            if has_lamp_mentioned or is_visible:
                return "stop"
            else:
                # 如果推理内容很短且只有stop，也允许停止（可能是模型直接输出stop）
                if len(text_lower.strip()) < 50 and "stop" in text_lower:
                    return "stop"
                print("🛡️ [审查驳回] 视觉证据不足，强制转向探测。")
                return "turn_left"

        if "backward" in action_part: return "move_backward"
        if "left" in action_part: return "turn_left"
        if "right" in action_part: return "turn_right"
        return "move_forward"

# ============================================================================
# 模块7: 决策 (Decision Making)
# ============================================================================
class DecisionModule:
    def __init__(self):
        self.decision_history = []
        self.action_space = ["move_forward", "turn_left", "turn_right", "stop", "move_backward"]
        self.last_image_gray = None # 用于存储上一帧的灰度图
        self.last_reasoning = None # 用于存储上一帧的推理
        self.same_reasoning_count = 0 # 推理重复计数

    def decide(self, llm_output: Dict, current_image: Image = None) -> str:
        action = llm_output.get("action", "move_forward")
        reasoning = llm_output.get("reasoning", "")
        
        # --- [策略介入] 推理重复检测 ---
        if self.last_reasoning is not None:
            # 计算推理相似度（简单的字符串比较）
            if reasoning == self.last_reasoning:
                self.same_reasoning_count += 1
                print(f"⚠️ [推理重复] 检测到相同推理，连续次数: {self.same_reasoning_count}")
                
                # 如果推理连续 5 次都一样，强制转向
                if self.same_reasoning_count >= 5:
                    print("🔄 [强制转向] 推理重复过多，执行 180° 转向！")
                    self.same_reasoning_count = 0
                    return "turn_left"  # 强制转向
            else:
                self.same_reasoning_count = 0
        
        self.last_reasoning = reasoning
        
        # --- [策略介入] 视觉停滞检测 ---
        if current_image is not None:
            # 将当前图转为灰度并缩小，计算差异
            curr_gray = np.array(current_image.convert('L').resize((64, 64)))
            
            if self.last_image_gray is not None:
                # 计算均方误差 (MSE)
                mse = np.mean((self.last_image_gray - curr_gray) ** 2)
                
                # 降低阈值：如果 MSE 极小（< 2.0），说明画面几乎没变
                if mse < 2.0:
                    print(f"🕵️ [决策干预] 检测到画面停滞 (MSE: {mse:.2f})，强制执行转向逃逸！")
                    action = random.choice(["turn_left", "turn_right"])
            
            self.last_image_gray = curr_gray # 更新缓存
        else:
            # 如果是旋转动作，重置缓存防止误差
            self.last_image_gray = None

        # 原有的逻辑：防止反复横跳
        if len(self.decision_history) >= 2:
            last_1 = self.decision_history[-1]['action']
            if last_1 == "move_backward" and action == "move_forward":
                action = random.choice(["turn_left", "turn_right"])

        self.decision_history.append({"action": action})
        return action

# ============================================================================
# 模块8: 执行 (Execution)_extract_action
# ============================================================================
class ExecutionModule:
    def __init__(self, simulator=None):
        self.simulator = simulator
    
    def execute(self, action: str) -> Dict[str, Any]:
        print(f"⚡ [执行模块] 正在执行: {action}")
        if self.simulator is None: return {"success": False}
        
        try:
            if action == "move_forward":
                # === [修改这里] 改回单步模式 (0.25m) ===
                self.simulator.step("move_forward")
                # self.simulator.step("move_forward") # <--- 注释掉或删除这行，不要连走两步！
            
            elif action == "move_backward":
                # 后退保持两步或者改成一步都可以，建议保持两步以便快速脱困
                self.simulator.step("move_backward")
                self.simulator.step("move_backward")
                
            elif action in ["turn_left", "turn_right"]:
                self.simulator.step(action)
            
            elif action == "stop":
                pass # 这里的 stop 是逻辑停止，不需要物理动作
                
            return {"success": True, "action": action}
        except Exception as e:
            print(f"❌ 执行错误: {e}")
            # 如果动作不存在 (比如打错字)，不要让程序崩，返回失败即可
            return {"success": False}

# ============================================================================
# 模块9: 循环 (Loop/Cycle)
# ============================================================================
class LoopController:
    """循环控制模块：管理循环"""
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.current_step = 0
        self.should_stop = False
    
    def should_continue(self) -> bool:
        if self.should_stop: return False
        if self.current_step >= self.max_steps: return False
        return True
    
    def advance_step(self):
        self.current_step += 1
        print(f"🔄 [循环控制] 第 {self.current_step}/{self.max_steps} 步")
    
    def stop(self):
        self.should_stop = True
        print("🛑 [循环控制] 收到停止信号")
    
    def reset(self):
        self.current_step = 0
        self.should_stop = False


# ============================================================================
# 模块10: 打分 (Scoring) - 稍微增强日志run
# ============================================================================
class ScoringModule:
    """打分模块：评估导航性能"""
    def __init__(self):
        self.scores = []
    
    def score(self, trajectory: List[Dict], goal_reached: bool = False,
              start_position: List[float] = None, 
              target_position: List[float] = None) -> Dict[str, Any]:
        
        # === SR (Success Rate) ===
        SR = 1.0 if goal_reached else 0.0
        
        # === 计算实际路径长度 ===
        actual_path_length = self._calculate_path_length(trajectory)
        
        # === 计算最优路径长度 (如果有真值目标) ===
        optimal_path_length = 0.0
        if start_position is not None and target_position is not None:
            # 这里用欧氏距离做近似，如果有 pathfinder 最好用 geodesic
            optimal_path_length = np.linalg.norm(np.array(start_position) - np.array(target_position))
        
        # === SPL ===
        if goal_reached and actual_path_length > 0 and optimal_path_length > 0:
            SPL = SR * (optimal_path_length / max(actual_path_length, optimal_path_length))
        else:
            SPL = 0.0
        
        score_result = {
            "SR": SR,
            "SPL": SPL,
            "goal_reached": goal_reached,
            "steps": len(trajectory),
            "path_length": actual_path_length
        }
        
        self.scores.append(score_result)
        if optimal_path_length > 0:
            print(f"   - 识别步数: {len(trajectory)}")
            print(f"   - 实际行走距离: {actual_path_length:.2f} 米")
            if goal_reached:
                print(f"   - 识别成功：代理找到台灯并停止")
            else:
                print(f"   - 识别失败：代理未找到台灯")
        return score_result
    
    def _calculate_path_length(self, trajectory: List[Dict]) -> float:
        total = 0.0
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]['perception']['spatial']['position']
            p2 = trajectory[i+1]['perception']['spatial']['position']
            if p1 is not None and p2 is not None:
                total += np.linalg.norm(np.array(p1) - np.array(p2))
        return total


# ============================================================================
# 核心集成: VLN Agent (整合所有模块)self.loop_controller
# ============================================================================
class VLNAgent:
    def __init__(self, scene_path: str, model_path: str):
        print("\n🚀 [System] 初始化 VLN Agent (距离增强版)...")
        self.simulator = self._init_simulator(scene_path)
        self.model, self.processor = self._init_llm(model_path)
        
        # 模块实例化
        self.instruction_mod = InstructionModule()
        self.planning_mod = PlanningModule()
        self.perception_mod = PerceptionModule(simulator=self.simulator)
        self.memory_mod = MemoryModule(max_memory_size=50)
        # 传递memory_module给CrossModalAlignmentModule
        self.alignment_mod = CrossModalAlignmentModule(
            model=self.model, 
            processor=self.processor, 
            memory_module=self.memory_mod
        )
        self.decision_mod = DecisionModule()
        self.execution_mod = ExecutionModule(simulator=self.simulator)
        self.loop_controller = LoopController(max_steps=50) # 修改最大步数为50
        self.scoring_mod = ScoringModule()
        
        # 路径查找器 (用于计算距离，不参与 Agent 决策)
        self.pathfinder = self.simulator.pathfinder
        
        # 状态变量
        self.verified = False  # 视角验证状态
        self.centered = False  # 居中调整状态
        self.approaching_lamp = False  # 是否正在接近lamp
        
        # 全局位置轨迹（用于智能探索）
        self.position_history = []  # 记录所有访问过的位置
        self.last_position = None  # 上一次位置
        self.position_stuck_count = 0  # 位置卡住计数
        self.unexplored_regions = []  # 未探索区域
        
        # 同向旋转控制
        self.last_turn_direction = None  # 上一次转向方向: "left" 或 "right"

    def scan_surroundings(self):
        print("🔄 [策略] 启动 360° 全景扫描...")
        views = []
        # 旋转 12 次，每次 30 度，覆盖 360 度
        for _ in range(12):
            self.simulator.step("turn_left")
            obs = self.simulator.get_sensor_observations()
            # 记录每一帧图像，后续可以拼接或让 LLM 批量处理
            views.append(self.perception_mod.perceive())
        
        # 将多张图拼接成一张长条图（全景图）传给 Qwen2-VL
        panorama = self._stitch_images([v["image"] for v in views])
        return panorama

    def _init_simulator(self, scene_path: str):
        print(f"🏗️ [System] 正在加载场景并配置动作空间...")
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.gpu_device_id = 0
        # === [修改点 1] 设置随机种子 ===
        sim_cfg.random_seed = random.randint(0, 1000000) 
        # =============================
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        # ... (后续传感器和动作空间配置保持不变)
        
        # 1. 配置传感器
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [480, 640]
        rgb_sensor.position = [0.0, 1.2, 0.0]
        agent_cfg.sensor_specifications = [rgb_sensor]
        depth_sensor = habitat_sim.CameraSensorSpec()
        depth_sensor.uuid = "depth_sensor"
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = [480, 640]
        depth_sensor.position = [0.0, 1.2, 0.0]
        agent_cfg.sensor_specifications.append(depth_sensor)
        semantic_sensor = habitat_sim.CameraSensorSpec()
        semantic_sensor.uuid = "semantic_sensor"
        semantic_sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor.resolution = [480, 640]
        semantic_sensor.position = [0.0, 1.2, 0.0]
        agent_cfg.sensor_specifications.append(semantic_sensor)
        # 2. === [核心修改] 显式定义动作空间 (加入后退) ===
        # Habitat 默认只有 forward/left/right，必须手动加 backward
        action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "stop": habitat_sim.agent.ActionSpec(
                "stop", habitat_sim.agent.ActuationSpec(amount=0.0)
            )
        }
        agent_cfg.action_space = action_space
        
        return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    def _init_llm(self, model_path: str):
        print(f"🧠 [System] 加载模型: {model_path}")
        
        base_model_path = "model_cache/qwen/Qwen2-VL-7B-Instruct"
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path, device_map="auto", quantization_config=bnb_config, trust_remote_code=True, local_files_only=True
        )
        
        model.load_adapter(model_path)
        print(f"✅ [System] LoRA 适配器已加载: {model_path}")
        
        processor = AutoProcessor.from_pretrained(base_model_path, min_pixels=256*28*28, max_pixels=1280*28*28, local_files_only=True)
        return model, processor

    def set_agent_state(self, position, rotation=None):
        """强制设置 Agent 位置"""
        agent = self.simulator.get_agent(0)
        state = habitat_sim.AgentState()
        state.position = position
        if rotation is not None:
            state.rotation = rotation
        agent.set_state(state)
        print(f"📍 [系统] Agent 已重置到坐标: {position}")
    
    def _extract_objects_from_reasoning(self, reasoning: str) -> List[str]:
        """从LLM推理中提取看到的物体"""
        objects = []
        reasoning_lower = reasoning.lower()
        
        # 定义常见物体关键词
        object_keywords = {
            'lamp': ['lamp', 'light', 'lampshade', 'lighting'],
            'table': ['table', 'desk', 'surface'],
            'chair': ['chair', 'seat', 'stool'],
            'sofa': ['sofa', 'couch', 'settee'],
            'bed': ['bed', 'mattress'],
            'door': ['door', 'doorway', 'entrance'],
            'window': ['window', 'glass'],
            'wall': ['wall', 'corner'],
            'floor': ['floor', 'ground'],
            'ceiling': ['ceiling'],
            'shelf': ['shelf', 'bookshelf', 'cabinet'],
            'plant': ['plant', 'flower', 'tree'],
            'picture': ['picture', 'painting', 'art', 'frame'],
            'tv': ['tv', 'television', 'screen'],
            'carpet': ['carpet', 'rug'],
        }
        
        # 检查推理中是否包含这些关键词
        for object_name, keywords in object_keywords.items():
            for keyword in keywords:
                if keyword in reasoning_lower:
                    if object_name not in objects:
                        objects.append(object_name)
                    break
        
        return objects
    
    def _get_lamp_position_in_view(self, reasoning: str) -> Optional[str]:
        """
        从推理文本中提取 lamp 在画面中的位置
        
        Returns:
            "left", "right", "center", 或 None
        """
        reasoning_lower = reasoning.lower()
        
        # 检测位置关键词
        left_keywords = ["left side", "left of", "on the left", "to the left"]
        right_keywords = ["right side", "right of", "on the right", "to the right"]
        center_keywords = ["center", "middle", "front", "directly ahead"]
        
        for keyword in center_keywords:
            if keyword in reasoning_lower:
                return "center"
        
        for keyword in left_keywords:
            if keyword in reasoning_lower:
                return "left"
        
        for keyword in right_keywords:
            if keyword in reasoning_lower:
                return "right"
        
        return None
    
    def _get_current_yaw(self) -> float:
        """
        获取当前朝向角度（度数）
        
        Returns:
            float: 当前朝向角度（0-360）
        """
        try:
            agent_state = self.simulator.get_agent(0).get_state()
            rotation = np.array(agent_state.rotation)
            
            # 如果是标量，直接返回
            if rotation.shape == ():
                yaw_degrees = float(rotation)
                if abs(yaw_degrees) > 10:
                    yaw_degrees = np.degrees(yaw_degrees)
                if yaw_degrees < 0:
                    yaw_degrees += 360
                if yaw_degrees >= 360:
                    yaw_degrees -= 360
                return yaw_degrees
            else:
                return 0.0
        except:
            return 0.0
    
    def _find_unexplored_position(self, current_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        找到未探索的区域
        
        Args:
            current_pos: 当前位置 [x, y, z]
        
        Returns:
            Optional[np.ndarray]: 未探索区域的位置，或 None
        """
        # 在当前位置周围生成候选点
        candidates = []
        
        # 在不同方向生成候选点（距离 2-4m）
        for dx in [-3, -2, 2, 3]:
            for dz in [-3, -2, 2, 3]:
                candidate = current_pos + np.array([dx, 0, dz])
                
                # 检查是否在场景边界内
                if -10 <= candidate[0] <= 10 and -10 <= candidate[2] <= 10:
                    # 检查是否访问过
                    rounded_candidate = (np.round(candidate[0] * 2) / 2, np.round(candidate[1] * 2) / 2, np.round(candidate[2] * 2) / 2)
                    
                    if rounded_candidate not in self.visited_positions:
                        candidates.append(candidate)
        
        # 如果有未探索的区域，返回距离最近的一个
        if candidates:
            distances = [np.linalg.norm(c - current_pos) for c in candidates]
            best_idx = np.argmin(distances)
            return candidates[best_idx]
        
        return None
    
    def _get_position_info(self, position: np.ndarray, rotation: np.ndarray) -> Dict[str, str]:
        """
        计算朝向信息
        
        Args:
            position: 位置坐标 [x, y, z]
            rotation: 旋转角度（弧度或度数）
        
        Returns:
            Dict: 包含朝向信息
        """
        # 提取位置坐标
        x, y, z = position[0], position[1], position[2]
        
        # 计算朝向（基于旋转角度）
        # rotation 可能是标量（弧度或度数）
        try:
            # 尝试转换为 numpy 数组
            rotation_array = np.array(rotation)
            
            # 调试信息：打印 rotation 的形状
            print(f"🔍 [调试] rotation 类型: {type(rotation)}, shape: {rotation_array.shape if hasattr(rotation_array, 'shape') else 'N/A'}, size: {rotation_array.size if hasattr(rotation_array, 'size') else 'N/A'}")
            
            # 检查是否是标量（shape 为空）
            if rotation_array.shape == ():
                # rotation 是标量，可能是 quaternion 对象
                # 简化处理：直接返回 0.0 作为默认值
                # TODO: 后续可以添加更精确的 quaternion 处理
                yaw_degrees = 0.0
                
                # 返回完整信息（不转换朝向描述）
                return {
                    'facing': f"{yaw_degrees:.1f}°",
                    'yaw': yaw_degrees,
                    'position': f"({x:.2f}, {y:.2f}, {z:.2f})"
                }
            
            # 检查数组形状
            elif rotation_array.size >= 4:
                # 获取四元数的 4 个分量
                if rotation_array.ndim == 1:
                    w, qx, qy, qz = rotation_array[0], rotation_array[1], rotation_array[2], rotation_array[3]
                elif rotation_array.ndim == 2 and rotation_array.shape[1] >= 4:
                    w, qx, qy, qz = rotation_array[0, 0], rotation_array[0, 1], rotation_array[0, 2], rotation_array[0, 3]
                elif rotation_array.ndim == 2 and rotation_array.shape[0] >= 4:
                    w, qx, qy, qz = rotation_array[0], rotation_array[1], rotation_array[2], rotation_array[3]
                else:
                    # 格式不对，返回默认值
                    return {
                        'facing': '未知',
                        'yaw': 0.0,
                        'position': f"({x:.2f}, {y:.2f}, {z:.2f})"
                    }
                
                # 计算 yaw 角度
                siny_cosp = 2 * (w * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = np.arctan2(siny_cosp, cosy_cosp)
                yaw_degrees = np.degrees(yaw)
                
                # 归一化到 0-360 度
                if yaw_degrees < 0:
                    yaw_degrees += 360
                
                # 转换为朝向描述
                if 337.5 <= yaw_degrees or yaw_degrees < 22.5:
                    facing = "北"
                elif 22.5 <= yaw_degrees < 67.5:
                    facing = "东北"
                elif 67.5 <= yaw_degrees < 112.5:
                    facing = "东"
                elif 112.5 <= yaw_degrees < 157.5:
                    facing = "东南"
                elif 157.5 <= yaw_degrees < 202.5:
                    facing = "南"
                elif 202.5 <= yaw_degrees < 247.5:
                    facing = "西南"
                elif 247.5 <= yaw_degrees < 292.5:
                    facing = "西"
                else:
                    facing = "西北"
                
                # 返回完整信息
                return {
                    'facing': facing,
                    'yaw': yaw_degrees,
                    'position': f"({x:.2f}, {y:.2f}, {z:.2f})"
                }
            else:
                # 数组大小不够，返回默认值
                return {
                    'facing': '未知',
                    'yaw': 0.0,
                    'position': f"({x:.2f}, {y:.2f}, {z:.2f})"
                }
        except Exception as e:
            # 如果转换失败，返回默认值
            print(f"⚠️ [朝向计算失败] 错误: {e}")
            return {
                'facing': '未知',
                'yaw': 0.0,
                'position': f"({x:.2f}, {y:.2f}, {z:.2f})"
            }
    
    def _evaluate_execution_result(self, action: str, dist_front: float, reasoning: str) -> str:
        """评估动作执行结果"""
        if action == "stop":
            # 如果是停止动作，检查是否真的找到了目标
            if any(keyword in reasoning.lower() for keyword in ["lamp", "target", "found"]):
                return "success"
            else:
                return "partial"
        elif action == "move_forward":
            # 如果是前进动作，检查是否成功移动
            if dist_front > 0.5:  # 如果前方距离正常，说明成功移动
                return "success"
            else:
                return "failure"  # 可能撞墙了
        elif "turn" in action:
            # 转向动作通常是成功的（避免撞墙）
            return "success"
        elif action == "move_backward":
            # 后退动作也是成功的（避免撞墙）
            return "success"
        else:
            return "partial"

    def run(self, instruction_text: str, target_pos: List[float] = None):
        print(f"\n{'='*20} 任务开始 (多视角校验 + 自主决策版) {'='*20}")
        save_dir = "guocheng"
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        else:
            for f in os.listdir(save_dir):
                if f.endswith(".jpg"): os.remove(os.path.join(save_dir, f))

        # === 创建日志文件（每次运行都覆盖）===
        log_file = os.path.join(save_dir, "navigation_log.txt")
        if os.path.exists(log_file):
            os.remove(log_file)  # 删除旧日志文件
        log_entries = []
        
        # 写入日志文件头
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Navigation Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Task: Find LAMP\n")
            f.write(f"{'='*60}\n")
        
        self.loop_controller.reset()
        self.memory_mod.memory.clear()
        
        # === 核心变量初始化 ===
        success = False 
        self.verified = False  # 视角验证状态位
        self.centered = False  # 居中调整状态位
        self.consecutive_wall_hits = 0  # 连续撞墙计数器
        self.last_wall_direction = None  # 上次撞墙的方向
        self.lamp_position = None  # lamp 位置记忆
        self.lamp_found_step = None  # 找到 lamp 的步数
        self.lamp_confirmed = False  # lamp 是否已确认
        
        # 已探索区域记忆
        self.visited_positions = []  # 记录访问过的位置
        self.current_region = None  # 当前区域
        self.region_visit_count = {}  # 区域访问计数
        self.same_view_count = 0  # 同一画面连续出现次数
        
        start_state = self.simulator.get_agent(0).get_state()
        start_pos = start_state.position
        prev_pos = np.array(start_pos)
        room_anchor_pos = np.array(start_pos) 
        
        while self.loop_controller.should_continue():
            current_step = self.loop_controller.current_step
            self.loop_controller.advance_step()
            
            # 1. 环境感知
            agent_state = self.simulator.get_agent(0).get_state()
            current_pos = np.array(agent_state.position)
            perception = self.perception_mod.perceive(agent_state)
            dist_front = perception.get("depth", 999.0)
            
            if perception["image"]:
                perception["image"].save(os.path.join(save_dir, f"step_{current_step:03d}.jpg"))
            
            # 2. 状态监测
            move_dist = np.linalg.norm(current_pos - prev_pos)
            is_stuck = (move_dist < 0.05 and current_step > 0)
            prev_pos = current_pos
            
            # 3. 大脑思考
            think_result = self.alignment_mod.think(
                perception=perception, 
                instruction=instruction_text,
                memory=self.memory_mod.retrieve(k=5),
                collision_warning=is_stuck, 
                step_count=current_step
            )
            
            reasoning = think_result.get("reasoning", "")
            reasoning_lower = reasoning.lower()
            action = self.decision_mod.decide(think_result, perception["image"])
            
            # === [重复探索检测] ===
            # 如果正在接近 lamp（推理中提到 lamp 或 visible），跳过重复探索检测
            is_approaching_lamp = any(w in reasoning_lower for w in ["lamp", "visible", "clearly visible"])
            
            # === [智能探索：位置变化检测] ===
            # 记录当前位置
            self.position_history.append(current_pos.copy())
            
            # 检测位置变化（x 或 z 变化小于 0.2）
            if self.last_position is not None:
                # 不是第一次循环，计算位置变化
                pos_change = np.linalg.norm(current_pos - self.last_position)
                
                # 如果没有看到 lamp 且位置变化小于 0.2
                if not is_approaching_lamp and pos_change < 0.2:
                    self.position_stuck_count += 1
                    print(f"🔍 [智能探索] 位置变化: {pos_change:.3f}m, 卡住计数: {self.position_stuck_count}")
                    
                    # 如果连续 2 次位置变化小于 0.2，前往未探索区域
                    if self.position_stuck_count >= 2:
                        print("🧭 [智能探索] 检测到卡住，前往未探索区域...")
                        
                        # 寻找未探索的区域
                        unexplored_pos = self._find_unexplored_position(current_pos)
                        
                        if unexplored_pos is not None:
                            # 计算转向角度
                            target_direction = unexplored_pos - current_pos
                            target_yaw = np.arctan2(target_direction[0], target_direction[2])
                            current_yaw = np.radians(self._get_current_yaw())
                            
                            # 计算需要转向的角度
                            turn_angle = target_yaw - current_yaw
                            if turn_angle > np.pi:
                                turn_angle -= 2 * np.pi
                            elif turn_angle < -np.pi:
                                turn_angle += 2 * np.pi
                            
                            # 转向到目标方向
                            turn_steps = int(abs(turn_angle) / (np.pi / 6))  # 30° = π/6
                            print(f"🧭 [智能探索] 转向 {np.degrees(turn_angle):.1f}° ({turn_steps} 步)...")
                            for _ in range(turn_steps):
                                if turn_angle > 0:
                                    self.execution_mod.execute("turn_left")
                                else:
                                    self.execution_mod.execute("turn_right")
                            
                            self.position_stuck_count = 0
                        else:
                            print("⚠️ [智能探索] 未找到未探索区域，随机转向...")
                            # 同向旋转：如果上次转向过，继续同方向
                            if self.last_turn_direction is not None:
                                turn_action = self.last_turn_direction
                                print(f"🔄 [智能探索] 同向旋转向 {turn_action} 转向...")
                                # 记录到日志
                                with open(log_file, 'a', encoding='utf-8') as f:
                                    f.write(f"🔄 [智能探索] 同向旋转向 {turn_action} 转向...\n")
                            else:
                                turn_action = "turn_left"
                                self.last_turn_direction = turn_action
                                print(f"🔄 [智能探索] 随机转向 {turn_action}...")
                                # 记录到日志
                                with open(log_file, 'a', encoding='utf-8') as f:
                                    f.write(f"🔄 [智能探索] 随机转向 {turn_action}...\n")
                            self.execution_mod.execute(turn_action)
                            self.position_stuck_count = 0
                else:
                    self.position_stuck_count = 0
            
            self.last_position = current_pos
            
            # === [重复探索检测] ===
            if not is_approaching_lamp:
                # 记录当前位置（四舍五入到 0.5m 精度）
                rounded_pos = (np.round(current_pos[0] * 2) / 2, np.round(current_pos[1] * 2) / 2, np.round(current_pos[2] * 2) / 2)
                
                # 检查是否访问过这个位置
                if rounded_pos in self.visited_positions:
                    self.same_view_count += 1
                    print(f"⚠️ [重复探索] 检测到已访问位置 {rounded_pos}，连续次数: {self.same_view_count}")
                    
                    # 如果同一位置连续出现超过 2 次，强制转向
                    if self.same_view_count >= 2:
                        print("🔄 [强制转向] 重复探索过多，执行 180° 转向！")
                        # 同向旋转：如果上次转向过，继续同方向
                        if self.last_turn_direction is not None:
                            turn_action = self.last_turn_direction
                            print(f"🔄 [强制转向] 同向旋转向 {turn_action} 转向...")
                            # 记录到日志
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"🔄 [强制转向] 同向旋转向 {turn_action} 转向...\n")
                        else:
                            turn_action = "turn_left"
                            self.last_turn_direction = turn_action
                            print(f"🔄 [强制转向] 随机转向 {turn_action}...")
                            # 记录到日志
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"🔄 [强制转向] 随机转向 {turn_action}...\n")
                        for _ in range(6):  # 180°
                            self.execution_mod.execute(turn_action)
                        self.same_view_count = 0
                        # 更新位置
                        agent_state = self.simulator.get_agent(0).get_state()
                        current_pos = np.array(agent_state.position)
                        prev_pos = current_pos
                else:
                    self.same_view_count = 0
                    self.visited_positions.append(rounded_pos)
                
                # 限制记忆大小，只保留最近 100 个位置
                if len(self.visited_positions) > 100:
                    self.visited_positions = self.visited_positions[-100:]

            # --- [长期记忆] 房间打转检测 ---
            if current_step % 30 == 0:
                dist_from_anchor = np.linalg.norm(current_pos - room_anchor_pos)
                if dist_from_anchor < 1.5: 
                    print(f"🧠 [长期记忆] 警告：已在当前区域逗留，强制逃逸。")
                    for _ in range(3): self.execution_mod.execute("turn_right")
                room_anchor_pos = current_pos 

            # 4. 视觉比重解析
            import re
            bbox_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', reasoning)
            area_ratio = 0.0
            if bbox_match:
                coords = [int(c) for c in bbox_match.groups()]
                area_ratio = ((coords[2] - coords[0]) * (coords[3] - coords[1])) / 1_000_000.0
            
            # 5. === [核心修改：多视角动态校验 + 居中调整] ===
            # 不再强制约束 'lamp' 关键字，只要模型想 stop，系统就启动校验
            print(f"🔍 [调试] action: {action}, verified: {self.verified}, centered: {self.centered}")
            
            if action == "stop":
                if not self.verified:
                    # 第一次确认：检查推理中是否明确说lamp可见
                    if "the lamp is visible in the current view" in reasoning_lower or "the lamp is clearly visible in the center of the image" in reasoning_lower:
                        # 直接向前走三步（每步0.75米）
                        print("🎯 [接近目标] Lamp已确认可见，直接向前走三步...")
                        for i in range(3):
                            # 获取新的感知
                            agent_state = self.simulator.get_agent(0).get_state()
                            perception = self.perception_mod.perceive(agent_state)
                            
                            # 保存图像
                            if perception["image"]:
                                perception["image"].save(os.path.join(save_dir, f"step_{current_step + i + 1:03d}.jpg"))
                            
                            # 获取位置和朝向信息
                            position_info = self._get_position_info(
                                np.array(agent_state.position),
                                np.array(agent_state.rotation)
                            )
                            
                            # 记录到日志
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"\n{'='*60}\n")
                                f.write(f"Step: {current_step + i + 1}\n")
                                f.write(f"{'-'*60}\n")
                                f.write(f"📍 位置: {position_info['position']}\n")
                                
                                # 1. 从推理中提取观测到的物体（不显示）
                                detected_objects = self._extract_objects_from_reasoning(reasoning)
                                
                                # 2. 推理过程
                                f.write(f"🧠 推理: {reasoning}\n")
                                
                                # 3. 执行动作
                                f.write(f"🎯 动作: move_forward\n")
                                f.write(f"{'='*60}\n")
                            
                            # 执行move_forward
                            self.execution_mod.execute("move_forward")
                            self.memory_mod.store(perception, "move_forward")
                        
                        # === [修复] 任务结束时记录 stop 作为结尾 ===
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\n{'='*60}\n")
                            f.write(f"Step: {current_step + 3}_end\n")
                            f.write(f"{'-'*60}\n")
                            f.write(f"🧠 推理: Task completed - target found and approached\n")
                            f.write(f"🎯 动作: stop (任务结束)\n")
                            f.write(f"{'='*60}\n")
                        
                        # === [修复] 二次确认成功后，标记任务成功并生成文件 ===
                        success = True
                        final_dist = 0.0
                        self.memory_mod.set_last_result("success")
                        print("🎯 [记忆增强] 任务成功！找到台灯并接近。")
                        self.memory_mod.export_memory(f"successful_navigation_step_{current_step}.json")
                        
                        # === 保存完整日志 ===
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\n{'='*60}\n")
                            f.write(f"{'='*60} 任务完成 {'='*60}\n")
                            f.write(f"成功状态: {success}\n")
                            f.write(f"识别步数: {current_step}\n")
                            f.write(f"{'='*60}\n")
                        
                        break  # 退出循环，结束任务
                    else:
                        # 记录 lamp 位置，不转向
                        print("🕵️ [主动观测] 疑似发现目标，记录位置并继续前进...")
                        self.verified = True
                        # 不执行转向，直接继续前进接近 lamp
                        # 这样可以避免 lamp 跑出视野
                        # 存储这一步动作并跳过本次循环，等待下一帧（新视角）的思考
                        self.memory_mod.store(perception, "move_forward")
                        continue
                elif self.verified and not self.centered:
                    # 第二次确认：先调整 lamp 到画面中央
                    lamp_position = self._get_lamp_position_in_view(reasoning)
                    
                    if lamp_position == "left":
                        print("🎯 [居中调整] Lamp 在左侧，向右调整...")
                        self.execution_mod.execute("turn_right")
                    elif lamp_position == "right":
                        print("🎯 [居中调整] Lamp 在右侧，向左调整...")
                        self.execution_mod.execute("turn_left")
                    elif lamp_position == "center":
                        print("✅ [居中调整] Lamp 已在中央，准备接近...")
                        self.centered = True
                    else:
                        print("⚠️ [居中调整] 无法确定 lamp 位置，默认居中...")
                        self.centered = True
                    
                    # 存储这一步动作并跳过本次循环
                    self.memory_mod.store(perception, action)
                    continue
            
            # 如果执行了移动或转向动作，重置校验状态
            # 注意：物理拦截（撞墙）不算真正的移动，不应该重置 verified
            if ("move" in action or "turn" in action) and dist_front >= 0.35:
                self.verified = False

            # 6. 物理避障拦截 (优先级最高)
            if action == "move_forward" and dist_front < 0.5:
                self.consecutive_wall_hits += 1
                print(f"🛡️ [物理拦截] 距离过近({dist_front:.2f}m)，连续撞墙 {self.consecutive_wall_hits} 次。")
                
                # 如果连续撞墙超过 3 次，强制执行大角度转向
                if self.consecutive_wall_hits >= 3:
                    print("🔄 [强制转向] 连续撞墙过多，执行 180° 转向逃逸！")
                    # 同向旋转：如果上次转向过，继续同方向
                    if self.last_turn_direction is not None:
                        turn_action = self.last_turn_direction
                        print(f"🔄 [物理拦截] 同向旋转向 {turn_action} 转向...")
                        # 记录到日志
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"🔄 [物理拦截] 同向旋转向 {turn_action} 转向...\n")
                    else:
                        turn_action = "turn_left"
                        self.last_turn_direction = turn_action
                        print(f"🔄 [物理拦截] 随机转向 {turn_action}...")
                        # 记录到日志
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"🔄 [物理拦截] 随机转向 {turn_action}...\n")
                    for _ in range(6):  # 6 * 30° = 180°
                        self.execution_mod.execute(turn_action)
                    self.consecutive_wall_hits = 0  # 重置计数器
                    action = "move_forward"  # 转向后继续前进
                else:
                    # 普通撞墙，随机转向
                    # 同向旋转：如果上次转向过，继续同方向
                    if self.last_turn_direction is not None:
                        action = self.last_turn_direction
                        print(f"🔄 [物理拦截] 同向旋转向 {action} 转向...")
                        # 记录到日志
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"🔄 [物理拦截] 同向旋转向 {action} 转向...\n")
                    else:
                        action = random.choice(["turn_left", "turn_right"])
                        self.last_turn_direction = action
                        print(f"🔄 [物理拦截] 随机转向 {action}...")
                        # 记录到日志
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"🔄 [物理拦截] 随机转向 {action}...\n")
                    # 避免连续向同一方向转向
                    if self.last_wall_direction == action:
                        action = "turn_right" if action == "turn_left" else "turn_left"
                    self.last_wall_direction = action
                # 不重置 verified，因为撞墙后位置没变，应该继续确认
                # self.verified = False  # 注释掉，避免死循环
            else:
                # 成功前进，重置撞墙计数器
                self.consecutive_wall_hits = 0

            # 7. 动作执行
            # 减少进门加速，只在无目标时应用
            # 只有明确说看到 lamp 时才认为有 lamp（排除 "no visible lamp" 等否定表达）
            has_lamp = any(w in reasoning_lower for w in ["lamp is visible", "lamp visible", "clearly visible lamp", "see a lamp", "found a lamp", "lamp on", "light is visible"])
            is_doorway = any(w in reasoning_lower for w in ["doorway", "opening", "enter"])
            
            # === [同向旋转强制] 在没有lamp的前提下，强制转向动作遵循同向旋转 ===
            if not has_lamp and action in ["turn_left", "turn_right"]:
                if self.last_turn_direction is not None:
                    # 强制改为同向旋转
                    original_action = action
                    action = self.last_turn_direction
                    print(f"🔄 [同向旋转强制] 原动作 {original_action} -> 改为 {action}")
                    # 记录到日志
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"🔄 [同向旋转强制] 原动作 {original_action} -> 改为 {action}\n")
                else:
                    # 记录这次转向方向
                    self.last_turn_direction = action
                    print(f"🔄 [同向旋转记录] 记录转向方向: {action}")
                    # 记录到日志
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"🔄 [同向旋转记录] 记录转向方向: {action}\n")
            
            # === [改进探索策略] 无目标时采用"随机转向 + 前进"组合 ===
            # 在接近lamp模式下不触发探索策略
            if not self.approaching_lamp and not has_lamp and action == "move_forward" and dist_front > 0.5:
                # 每 3 步强制转向一次，避免一直走直线
                if current_step % 3 == 0 and current_step > 0:
                    # 同向旋转：如果上次转向过，继续同方向
                    if self.last_turn_direction is not None:
                        action = self.last_turn_direction
                        print(f"🔄 [探索策略] 无目标，同向旋转向 {action} 转向...")
                        # 记录到日志
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"🔄 [探索策略] 无目标，同向旋转向 {action} 转向...\n")
                    else:
                        action = random.choice(["turn_left", "turn_right"])
                        self.last_turn_direction = action
                        print(f"🔄 [探索策略] 无目标，随机转向 {action}...")
                        # 记录到日志
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"🔄 [探索策略] 无目标，随机转向 {action}...\n")
                    self.execution_mod.execute(action)
                    # 转向后继续前进
                    self.execution_mod.execute("move_forward")
                elif is_doorway and dist_front > 1.2:
                    print("🚀 [进门增强] 无目标时加速通过门口...")
                    self.execution_mod.execute("move_forward")
                    self.execution_mod.execute("move_forward")
                else:
                    self.execution_mod.execute(action)
            else:
                # 如果有lamp或正在接近lamp，重置转向方向
                if has_lamp or self.approaching_lamp:
                    self.last_turn_direction = None
                self.execution_mod.execute(action)
            
            # === 记录日志 ===
            # 只记录模型的推理和动作，不包含提示词
            # 获取方位和朝向信息
            agent_state = self.simulator.get_agent(0).get_state()
            position_info = self._get_position_info(
                np.array(agent_state.position),
                np.array(agent_state.rotation)
            )
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Step: {current_step}\n")
                f.write(f"{'-'*60}\n")
                f.write(f"📍 位置: {position_info['position']}\n")
                f.write(f"👁️ Yaw: {position_info['yaw']:.1f}°\n")
                
                # 1. 从推理中提取观测到的物体（不显示）
                detected_objects = self._extract_objects_from_reasoning(reasoning)
                
                # 2. 推理过程
                f.write(f"🧠 推理: {reasoning}\n")
                
                # 3. 执行动作（记录修改后的动作）
                f.write(f"🎯 动作: {action}\n")
                f.write(f"{'='*60}\n")
            
            # === [记忆增强] 执行结果评估和记忆存储 ===
            execution_result = self._evaluate_execution_result(action, dist_front, reasoning_lower)
            
            # 设置上一步的结果用于重要性评估
            self.memory_mod.set_last_result(execution_result)
            
            # 存储记忆并获取物体关联信息
            memory_result = self.memory_mod.store(perception, action)
            associations = memory_result.get('associations', [])
            
            # === 记录物体关联信息到日志 ===
            if associations:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"🔗 [物体关联] 本步识别到 {len(associations)} 个物体:\n")
                    for assoc in associations:
                        if assoc['is_new']:
                            f.write(f"  ➕ 新物体: {assoc['object_type']} (ID: {assoc['object_id']})\n")
                        else:
                            dist_str = f", 距离上次位置: {assoc['distance']:.2f}m" if assoc['distance'] else ""
                            f.write(f"  🔄 已知物体: {assoc['object_type']} (ID: {assoc['object_id']}){dist_str}\n")
                    f.write(f"{'='*60}\n")
            
            # 定期记忆巩固
            if current_step % 50 == 0 and current_step > 0:
                self.memory_mod.consolidate_memory()
                print(f"🧠 [记忆巩固] 第 {current_step} 步执行记忆巩固")
                
                # 在新的作用域中写入日志
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"🧠 [记忆巩固] 第 {current_step} 步执行记忆巩固\n")
                    f.write(f"{'='*60}\n")

        print(f"\n{'='*20} 任务结束 {'='*20}")
        self.scoring_mod.score(
            trajectory=self.memory_mod.get_trajectory(),
            goal_reached=success,
            start_position=start_pos,
            target_position=target_pos
        )


# ============================================================================
# Main Entry Point (核心修改逻辑)
# ============================================================================
if __name__ == "__main__":
    # 1. [修改] 场景路径换成现代公寓 (确保你有这个文件)
    SCENE_FILE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    MODEL_PATH = "saves/qwen2vl-7b-vln/lora/sft"  # 使用微调后的 LoRA 模型
    
    if not os.path.exists(SCENE_FILE):
        print(f"❌ 找不到 apartment_1，尝试使用 apartment_0...")
        SCENE_FILE = "data/scene_datasets/habitat-test-scenes/apartment_0.glb"
        if not os.path.exists(SCENE_FILE):
            raise FileNotFoundError(f"❌ 找不到地图文件: {SCENE_FILE}")
    
    try:
        agent = VLNAgent(scene_path=SCENE_FILE, model_path=MODEL_PATH)
        sim = agent.simulator
        pathfinder = sim.pathfinder
        
        # 2. [修改] 使用固定起点，随机朝向
        print("🎲 [System] 使用固定起点，随机朝向...")
        # 固定起点坐标
        start_pos = [4.5, -0.8, 0.7]
        
        # 随机朝向（0 到 2π 弧度）
        random_yaw = random.uniform(0, 2 * np.pi)
        rotation = R.from_euler('YXZ', [random_yaw, 0, 0]).as_quat()
        
        agent.set_agent_state(start_pos, rotation)
        print(f"📍 起点坐标: {start_pos}")
        print(f"🧭 随机朝向: {random_yaw:.2f} 弧度 ({np.degrees(random_yaw):.1f}°)")

        # 3. [简化] 不设置目标点，让代理自由探索直到找到台灯
        target_pos = None
        print(f"📏 [System] 任务：探索环境直到找到台灯")
        
        # 4. 运行任务
        instruction = "Find the LAMP."
        agent.run(instruction, target_pos=target_pos)
        draw_trajectory(agent, target_pos=target_pos, save_path="vln_result_with_goal.png")
        
    except KeyboardInterrupt:
        print("\n👋 用户中断程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()