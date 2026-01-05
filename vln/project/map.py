import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_semantic_topdown_map(scene_path, save_path="semantic_map_vln.png"):
    """
    加载测试场景并输出带有语义（如果可用）的二维地图
    """
    # 1. 模拟器配置
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    
    # 对于 apartment_1.glb，物理通常是关闭的以加快加载
    cfg = habitat_sim.Configuration(backend_cfg, [habitat_sim.AgentConfiguration()])
    sim = habitat_sim.Simulator(cfg)

    # 2. 获取 Top-down Map (占据地图)
    # 找到一个可导航点作为高度参考
    ref_point = sim.pathfinder.get_random_navigable_point()
    meters_per_pixel = 0.05  # 5cm 精度
    tdm = sim.pathfinder.get_topdown_view(meters_per_pixel, ref_point[1])

    # 3. 准备绘图
    plt.figure(figsize=(12, 12))
    # 绘制黑白地图底图
    plt.imshow(tdm, cmap="Greys", origin="lower")
    
    # 4. 坐标投影转换设置
    bounds = sim.pathfinder.get_bounds()
    min_bound = bounds[0]

    # 5. 语义提取尝试
    scene = sim.semantic_scene
    targets = {"lamp": "gold", "sofa": "blue", "table": "green", "chair": "red"}
    found_any = False

    print(f"\n{'='*15} 场景数据扫描 {'='*15}")
    
    if scene and len(scene.objects) > 0:
        for obj in scene.objects:
            raw_name = obj.category.name().lower()
            for target_key, color in targets.items():
                if target_key in raw_name:
                    pos = obj.aabb.center
                    grid_x = (pos[0] - min_bound[0]) / meters_per_pixel
                    grid_z = (pos[2] - min_bound[2]) / meters_per_pixel
                    
                    plt.scatter(grid_x, grid_z, c=color, s=200, edgecolors='black', 
                                label=target_key if target_key not in [l.get_label() for l in plt.gca().get_lines()] else "",
                                zorder=5)
                    print(f"✅ 发现语义目标 [{target_key.upper()}]: ({pos[0]:.2f}, {pos[2]:.2f})")
                    found_any = True
    else:
        print("ℹ️ 该场景文件不包含语义实例数据 (apartment_1.glb 常见现象)")
        print("💡 建议：请结合 test.py 输出的 target_pos 在图中手动对应。")

    # 6. 保存结果
    plt.title(f"Top-down Map Prior\nScene: {os.path.basename(scene_path)}")
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # 处理图例
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    sim.close()
    print(f"\n📍 地图已生成: {save_path}")

# ==========================================
# 路径配置 (已根据你的要求更新)
# ==========================================
scene_file = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"

if __name__ == "__main__":
    if os.path.exists(scene_file):
        generate_semantic_topdown_map(scene_file)
    else:
        print(f"❌ 依然找不到文件: {scene_file}")
        print("请尝试使用绝对路径，例如: /home/ubuntu/YuanNav/vln/project/" + scene_file)
        