# VLN Agent - Vision-Language Navigation Agent

一个基于多模态大语言模型（Qwen2-VL）的视觉-语言导航（VLN）智能体，具有增强的记忆系统和自主决策能力。

## 功能特点

### 核心功能
- **多模态感知**：结合RGB图像、深度信息和空间位置进行环境感知
- **记忆增强**：多层次记忆系统，支持短期、中期和长期记忆
- **自主决策**：基于视觉-语言对齐的智能决策机制
- **路径规划**：结合记忆和实时感知的路径规划算法

### 智能探索策略
- **同向旋转**：避免反复横跳，保持探索方向一致性
- **智能避障**：基于深度信息的物理避障机制
- **区域探索**：记录已探索区域，避免重复探索
- **房间打转检测**：检测并避免在原地打转

### 目标检测与追踪
- **多视角校验**：通过多次视角确认目标物体
- **居中调整**：自动调整目标到画面中央
- **接近策略**：找到目标后自动接近

## 环境要求

### Python版本
- Python >= 3.9

### 主要依赖
- `habitat-sim` >= 0.2.0
- `torch` >= 2.0.0
- `transformers` >= 4.30.0
- `numpy` >= 1.24.0
- `opencv-python` >= 4.8.0
- `Pillow` >= 10.0.0
- `scipy` >= 1.11.0
- `matplotlib` >= 3.7.0
- `bitsandbytes` >= 0.41.0

## 安装

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/YuanNav.git
cd YuanNav
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 下载模型
```bash
# 下载基础模型
mkdir -p model_cache/qwen
cd model_cache/qwen
git clone https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

# 下载LoRA模型（如果有）
cd ../..
python -c "from transformers import Qwen2VLForConditionalGeneration; model = Qwen2VLForConditionalGeneration.from_pretrained('path/to/lora/model')"
```

### 5. LoRA模型加载说明

本项目使用LoRA（Low-Rank Adaptation）进行模型微调。加载LoRA模型的步骤如下：

#### 5.1 模型加载代码
在`test.py`的`_init_llm`方法中，LoRA模型的加载逻辑如下：

```python
def _init_llm(self, model_path: str):
    print(f"🧠 [System] 加载模型: {model_path}")
    
    # 1. 加载基础模型（4bit量化）
    base_model_path = "model_cache/qwen/Qwen2-VL-7B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path, 
        device_map="auto", 
        quantization_config=bnb_config, 
        trust_remote_code=True, 
        local_files_only=True
    )
    
    # 2. 加载LoRA适配器
    model.load_adapter(model_path)
    print(f"✅ [System] LoRA 适配器已加载: {model_path}")
    
    # 3. 加载处理器
    processor = AutoProcessor.from_pretrained(
        base_model_path, 
        min_pixels=256*28*28, 
        max_pixels=1280*28*28, 
        local_files_only=True
    )
    
    return model, processor
```

#### 5.2 LoRA模型路径配置
在`test.py`的主程序中，配置LoRA模型路径：

```python
# LoRA模型路径
MODEL_PATH = "saves/qwen2vl-7b-vln/lora/sft"

# 检查LoRA模型是否存在
if not os.path.exists(MODEL_PATH):
    print(f"❌ 找不到LoRA模型: {MODEL_PATH}")
    print(f"请先训练LoRA模型或下载预训练的LoRA权重")
    exit(1)
```

#### 5.3 LoRA模型训练（可选）
如果需要训练自己的LoRA模型，可以使用以下配置：

```bash
# 使用YuanNav训练LoRA
python src/llamafactory/cli.py \
    examples/train_lora/qwen2_5vl_lora_sft.yaml \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --dataset vln_navigation_demo \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --output_dir saves/qwen2vl-7b-vln/lora/sft
```

#### 5.4 LoRA模型结构
训练后的LoRA模型目录结构：

```
saves/qwen2vl-7b-vln/lora/sft/
├── adapter_config.json       # LoRA配置文件
├── adapter_model.safetensors # LoRA权重文件
├── README.md               # 训练说明
├── merges.txt              # tokenizer合并文件
├── tokenizer.json           # tokenizer配置
└── vocab.json             # 词汇表
```

#### 5.5 验证LoRA加载
运行程序时，如果LoRA加载成功，会看到以下输出：

```
🧠 [System] 加载模型: saves/qwen2vl-7b-vln/lora/sft
✅ [System] LoRA 适配器已加载: saves/qwen2vl-7b-vln/lora/sft
```

#### 5.6 常见问题

**Q: 如何更换LoRA模型？**
A: 修改`test.py`中的`MODEL_PATH`变量，指向新的LoRA模型目录。

**Q: 如何使用基础模型（不加载LoRA）？**
A: 注释掉`model.load_adapter(model_path)`这一行，直接使用基础模型。

**Q: LoRA模型加载失败怎么办？**
A: 检查以下几点：
1. LoRA模型路径是否正确
2. `adapter_config.json`和`adapter_model.safetensors`文件是否存在
3. 基础模型和LoRA模型是否兼容（相同的基础模型）

**Q: 如何合并LoRA到基础模型？**
A: 使用以下代码合并：
```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model")
```

### 6. 准备场景数据
```bash
# 下载Habitat场景
mkdir -p data/scene_datasets/habitat-test-scenes
# 将场景文件（.glb）放到这个目录
```

## 使用方法

### 基本使用
```bash
cd vln/project
python test.py
```

### 自定义配置
```python
from vln.project.test import VLNAgent

# 创建agent
agent = VLNAgent(
    scene_path="data/scene_datasets/habitat-test-scenes/apartment_1.glb",
    model_path="saves/qwen2vl-7b-vln/lora/sft"
)

# 设置起点和朝向
import numpy as np
from scipy.spatial.transform import Rotation as R

start_pos = [4.5, -0.8, 0.7]
random_yaw = np.random.uniform(0, 2 * np.pi)
rotation = R.from_euler('YXZ', [random_yaw, 0, 0]).as_quat()

agent.set_agent_state(start_pos, rotation)

# 运行任务
agent.run("Find the LAMP.")

# 绘制轨迹
agent.draw_trajectory(save_path="trajectory.png")
```

## 项目结构

```
vln/project/
├── test.py                      # 主程序
├── enhanced_memory_system.py      # 增强记忆系统
├── map.py                       # 地图相关功能
└── README.md                    # 本文件
```

## 核心模块

### 1. InstructionModule（指令模块）
- 解析导航指令
- 管理指令历史

### 2. PlanningModule（规划模块）
- 生成导航计划
- 路径规划

### 3. PerceptionModule（感知模块）
- RGB图像感知
- 深度信息获取
- 空间位置计算

### 4. MemoryModule（记忆模块）
- 短期记忆（最近10步）
- 中期记忆（重要经验）
- 长期记忆（成功/失败模式）

### 5. CrossModalAlignmentModule（跨模态对齐模块）
- 视觉-语言对齐
- 多模态推理

### 6. DecisionModule（决策模块）
- 动作选择
- 策略介入（重复检测、画面停滞检测）

### 7. ExecutionModule（执行模块）
- 动作执行
- 物理避障

### 8. LoopController（循环控制模块）
- 步数控制
- 循环管理

### 9. ScoringModule（打分模块）
- 成功率计算
- 路径长度评估

### 10. EnhancedMemorySystem（增强记忆系统）
- 记忆类型分类
- 记忆重要性评分
- 记忆检索和更新

## 配置说明

### 场景配置
```python
# 在test.py中修改场景路径
SCENE_FILE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
```

### 模型配置
```python
# 在test.py中修改模型路径
MODEL_PATH = "saves/qwen2vl-7b-vln/lora/sft"
```

### 起点配置
```python
# 在test.py中修改起点
start_pos = [4.5, -0.8, 0.7]  # [x, y, z]
```

### 最大步数配置
```python
# 在LoopController中修改
self.loop_controller = LoopController(max_steps=50)
```

## 输出说明

### 日志文件
- `guocheng/navigation_log.txt` - 导航过程日志
- `successful_navigation_step_*.json` - 成功导航的记忆导出

### 图像文件
- `guocheng/step_*.jpg` - 每步的视角图像
- `vln_path_pro.png` - 轨迹可视化图

### 统计信息
- 成功率
- 识别步数
- 路径长度

## 性能优化

### GPU加速
```bash
# 使用CUDA
export CUDA_VISIBLE_DEVICES=0
python test.py

# 使用多GPU
export CUDA_VISIBLE_DEVICES=0,1
python test.py
```

### 量化加速
```python
# 使用4bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```



- Habitat-sim团队提供仿真环境
- Qwen团队提供多模态模型
