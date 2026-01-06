# VLN Agent - Vision-Language Navigation Agent

一个基于多模态大语言模型（Qwen2-VL）的视觉-语言导航agent（VLN），具有增强的记忆系统和自主决策能力。

![Uploading image.png…]()

## 功能特点

### 核心功能
- **多模态感知**：结合RGB图像、深度信息和Instruction进行环境感知
- **记忆增强**：多层次记忆系统，支持短期、中期和长期记忆
- **路径规划**：结合记忆和实时感知的路径规划算法

### 智能探索策略
- **智能避障**：基于深度信息的物理避障机制
- **区域探索**：记录已探索区域，避免重复探索
- **房间打转检测**：检测并避免在原地打转

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

### 3. 下载模型
```bash
# 下载基础模型
mkdir -p model_cache/qwen
cd model_cache/qwen
git clone https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

# 下载LoRA模型（如果有）
cd ../..
python -c "from transformers import Qwen2VLForConditionalGeneration; model = Qwen2VLForConditionalGeneration.from_pretrained('path/to/lora/model')"
```

- GitHub: https://github.com/fuu9775-ui/VLN-YuanNav
- Email: cjz18936691230@163.com


For full details, please refer to the README file in the project directory.
