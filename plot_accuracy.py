import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_accuracy_curve(accuracy_file, save_path=None, auto_range=True, y_min=None, y_max=None):
    """
    绘制准确率曲线图
    
    Args:
        accuracy_file: validation_accuracy.txt文件路径
        save_path: 保存图片的路径，如果为None则显示图片
        auto_range: 是否自动调整坐标轴范围
        y_min: 手动设置Y轴最小值
        y_max: 手动设置Y轴最大值
    """
    
    # 读取数据
    epochs = []
    accuracies = []
    
    with open(accuracy_file, 'r') as f:
        lines = f.readlines()
        # 跳过表头（第一行）
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    epoch, acc = parts[0], parts[1]
                    epochs.append(int(epoch))
                    accuracies.append(float(acc))
    
    if len(epochs) == 0:
        print("错误: 没有读取到有效数据")
        return None, None
    
    print(f"读取到 {len(epochs)} 个epoch的数据")
    print(f"准确率范围: {min(accuracies):.2f}% - {max(accuracies):.2f}%")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制准确率曲线
    ax.plot(epochs, accuracies, 
            color='#2E86AB',      # 线条颜色（蓝色）
            linewidth=2,           # 线宽
            marker='o',            # 标记点形状
            markersize=4,          # 标记点大小
            markerfacecolor='white',  # 标记点填充颜色
            markeredgewidth=1.5,      # 标记点边框宽度
            markeredgecolor='#2E86AB', # 标记点边框颜色
            label='DualCascadeTSF-MobileNetV2')
    
    # 设置坐标轴范围
    ax.set_xlim(0, max(epochs) + 5)
    
    if auto_range:
        # 自动调整Y轴范围，留出5%的边距
        y_min_auto = min(accuracies) - (max(accuracies) - min(accuracies)) * 0.05
        y_max_auto = max(accuracies) + (max(accuracies) - min(accuracies)) * 0.05
        ax.set_ylim(y_min_auto, y_max_auto)
    else:
        # 使用手动设置的范围
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if y_max is not None:
            ax.set_ylim(top=y_max)
    
    # 设置坐标轴刻度
    ax.set_xticks(np.arange(0, max(epochs) + 1, max(1, max(epochs) // 10)))
    
    # 动态设置Y轴刻度
    y_min_val = y_min if y_min is not None else min(accuracies)
    y_max_val = y_max if y_max is not None else max(accuracies)
    y_range = y_max_val - y_min_val
    y_step = max(1, y_range // 5)  # 大约5个刻度
    y_ticks = np.arange(np.floor(y_min_val / y_step) * y_step, 
                        np.ceil(y_max_val / y_step) * y_step + y_step, 
                        y_step)
    ax.set_yticks(y_ticks)
    
    # 设置坐标轴标签
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    
    # 设置标题
    ax.set_title('Accuracy curves on the Hockey Fights dataset', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 显示网格
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 设置图例
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # 设置坐标轴边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # 在最高点添加标注
    max_acc = max(accuracies)
    max_epoch = epochs[accuracies.index(max_acc)]
    ax.annotate(f'Best: {max_acc:.2f}%', 
                xy=(max_epoch, max_acc),
                xytext=(max_epoch - 10, max_acc + 1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                fontsize=10,
                color='red')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()
    
    return epochs, accuracies


def plot_accuracy_curve_fixed(accuracy_file, save_path=None):
    """
    修复版：确保所有准确率都能显示
    使用固定的Y轴范围，但根据实际数据调整
    """
    
    # 读取数据
    epochs = []
    accuracies = []
    
    with open(accuracy_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        epoch = int(parts[0])
                        acc = float(parts[1])
                        epochs.append(epoch)
                        accuracies.append(acc)
                    except ValueError:
                        print(f"跳过无效行: {line}")
                        continue
    
    if len(epochs) == 0:
        print("错误: 没有读取到有效数据")
        return
    
    print(f"数据统计:")
    print(f"  - Epoch范围: {min(epochs)} - {max(epochs)}")
    print(f"  - 准确率范围: {min(accuracies):.2f}% - {max(accuracies):.2f}%")
    print(f"  - 平均准确率: {np.mean(accuracies):.2f}%")
    print(f"  - 最佳准确率: {max(accuracies):.2f}% (Epoch {epochs[accuracies.index(max(accuracies))]})")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制准确率曲线
    ax.plot(epochs, accuracies, 
            color='#2E86AB',
            linewidth=2.5,
            marker='o',
            markersize=5,
            markerfacecolor='white',
            markeredgewidth=1.5,
            markeredgecolor='#2E86AB',
            label='DualCascadeTSF-MobileNetV2',
            linestyle='-')
    
    # 设置坐标轴范围 - 关键修复
    ax.set_xlim(0, max(epochs) + 5)
    
    # 动态计算Y轴范围，确保所有点都可见
    y_padding = (max(accuracies) - min(accuracies)) * 0.1
    ax.set_ylim(min(accuracies) - y_padding, max(accuracies) + y_padding)
    
    # 设置刻度
    x_ticks = np.arange(0, max(epochs) + 1, max(1, max(epochs) // 10))
    ax.set_xticks(x_ticks)
    
    # 设置Y轴刻度，确保覆盖所有准确率
    y_min_floor = np.floor(min(accuracies) / 5) * 5
    y_max_ceil = np.ceil(max(accuracies) / 5) * 5
    y_ticks = np.arange(y_min_floor, y_max_ceil + 5, 5)
    ax.set_yticks(y_ticks)
    
    # 设置标签
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy curves on the Hockey Fights dataset', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 网格
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 图例
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # 标注最佳点
    best_epoch = epochs[accuracies.index(max(accuracies))]
    best_acc = max(accuracies)
    ax.plot(best_epoch, best_acc, 'ro', markersize=8, markeredgecolor='red', markerfacecolor='red')
    ax.annotate(f'Best: {best_acc:.2f}%', 
                xy=(best_epoch, best_acc),
                xytext=(best_epoch - 15, best_acc + 1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=11,
                color='red',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
    
    # 添加最终准确率标注
    final_epoch = epochs[-1]
    final_acc = accuracies[-1]
    ax.annotate(f'Final: {final_acc:.2f}%', 
                xy=(final_epoch, final_acc),
                xytext=(final_epoch - 15, final_acc - 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=11,
                color='green',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8))
    
    # 设置背景色
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()
    
    return epochs, accuracies


def check_data_file(accuracy_file):
    """检查数据文件内容"""
    print(f"\n检查文件: {accuracy_file}")
    print("-" * 50)
    
    if not os.path.exists(accuracy_file):
        print(f"文件不存在: {accuracy_file}")
        return False
    
    with open(accuracy_file, 'r') as f:
        lines = f.readlines()
    
    print(f"文件总行数: {len(lines)}")
    print(f"表头: {lines[0].strip()}")
    
    # 显示前10行数据
    print("\n前10行数据:")
    for i, line in enumerate(lines[1:11]):
        if line.strip():
            print(f"  {i+1}: {line.strip()}")
    
    # 显示后10行数据
    print("\n后10行数据:")
    for line in lines[-10:]:
        if line.strip():
            print(f"  {line.strip()}")
    
    return True


# ============ 使用示例 ============

if __name__ == '__main__':
    
    # 方法1：检查数据文件
    accuracy_file = '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/TSMBT_RWF-2000_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt'
    
    # 先检查文件内容
    if check_data_file(accuracy_file):
        # 使用修复版绘图函数
        print("\n开始绘制准确率曲线...")
        epochs, accuracies = plot_accuracy_curve_fixed(
            accuracy_file, 
            save_path='accuracy_curve_CrowdViolence_fixed01.png'
        )
        
        # 也可以使用自动调整版本#HockeyFights#CrowdViolence#RWF-2000
        # epochs, accuracies = plot_accuracy_curve(
        #     accuracy_file, 
        #     save_path='accuracy_curve_CrowdViolence.png',
        #     auto_range=True
        # )
    else:
        print("请检查文件路径是否正确")
        
        # 尝试查找文件
        print("\n正在搜索validation_accuracy.txt文件...")
        os.system(f'find /mnt/data-windows/jjz -name "validation_accuracy.txt" 2>/dev/null')
    
    # 方式2：如果有多个模型的结果，可以绘制对比图
    # 示例：
    # model_files = [
    #     accuracy_files = [
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/DTSF1_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/DTSF2_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/DTSF3_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/DTSF4_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/DTSF5_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
    # ]
    # model_names = [
    #     'DualCascadeTSF-MobileNetV2',
    #     'TSM-MobileNetV2', 
    #     'TSF-MobileNetV2'
    # ]
    # plot_multiple_curves(model_files, model_names, save_path='comparison_curves.png')