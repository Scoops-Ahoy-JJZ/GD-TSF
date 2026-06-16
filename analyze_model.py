import matplotlib.pyplot as plt
import numpy as np
import os
import re

def parse_accuracy_file(file_path):
    """
    解析准确率文件，支持多种格式
    返回: (epochs, accuracies, label_name)
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None, None, None
    
    epochs = []
    accuracies = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 自动检测文件格式
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 格式1: "epoch X: accuracy = Y" 或 "Epoch X, accuracy: Y"
        match = re.search(r'(?:epoch|Epoch)\s*(\d+)[\s:,]+(?:accuracy|acc|val_acc)[\s:=]+([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            acc = float(match.group(2))
            epochs.append(epoch)
            accuracies.append(acc)
            continue
        
        # 格式2: "X Y" (两列数据)
        parts = line.split()
        if len(parts) >= 2:
            try:
                epoch = int(parts[0])
                acc = float(parts[1])
                epochs.append(epoch)
                accuracies.append(acc)
            except ValueError:
                continue
        
        # 格式3: 纯数字 (假设每行一个准确率值)
        try:
            acc = float(line)
            epochs.append(len(accuracies) + 1)
            accuracies.append(acc)
        except ValueError:
            pass
    
    # 从路径中提取数据集名称作为标签
    if 'CrowdViolence' in file_path:
        label = 'CrowdViolence'
    elif 'HockeyFights' in file_path:
        label = 'HockeyFights'
    elif 'RWF-2000' in file_path:
        label = 'RWF-2000'
    else:
        # 使用文件名作为标签
        label = os.path.basename(os.path.dirname(file_path))
    
    return epochs, accuracies, label

def plot_multiple_accuracy_curves(file_paths, save_path='accuracy_curves.png', 
                                  title='Model Accuracy Comparison', 
                                  figsize=(10, 6), 
                                  auto_range=True,
                                  show_grid=True,
                                  markers=['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']):
    """
    绘制多条准确率曲线
    
    参数:
        file_paths: 文件路径列表，每个元素可以是字符串路径，或(路径, 自定义标签)元组
        save_path: 保存图片的路径
        title: 图表标题
        figsize: 图片大小
        auto_range: 是否自动调整y轴范围
        show_grid: 是否显示网格
        markers: 标记样式列表
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
    
    all_data = []
    max_epoch = 0
    min_acc = 1.0
    max_acc = 0.0
    
    for idx, file_info in enumerate(file_paths):
        # 支持元组格式：(文件路径, 自定义标签)
        if isinstance(file_info, tuple):
            file_path = file_info[0]
            custom_label = file_info[1]
        else:
            file_path = file_info
            custom_label = None
        
        epochs, accuracies, auto_label = parse_accuracy_file(file_path)
        
        if epochs is None or len(epochs) == 0:
            print(f"跳过无效文件: {file_path}")
            continue
        
        # 使用自定义标签或自动生成的标签
        label = custom_label if custom_label else auto_label
        
        # 绘制曲线
        marker = markers[idx % len(markers)]
        color = colors[idx]
        
        plt.plot(epochs, accuracies, 
                marker=marker, 
                markersize=4,
                linewidth=2,
                color=color,
                label=label,
                markevery=max(1, len(epochs)//20))  # 自动控制标记密度
        
        all_data.append((epochs, accuracies, label))
        max_epoch = max(max_epoch, max(epochs))
        min_acc = min(min_acc, min(accuracies))
        max_acc = max(max_acc, max(accuracies))
        
        # 显示最佳准确率
        best_acc = max(accuracies)
        best_epoch = epochs[np.argmax(accuracies)]
        print(f"{label}: 最佳准确率 = {best_acc:.4f} (Epoch {best_epoch})")
    
    if not all_data:
        print("没有有效的文件可以绘制！")
        return None
    
    # 设置坐标轴范围
    if auto_range:
        # 自动调整范围，留出一些边距
        y_margin = (max_acc - min_acc) * 0.05
        plt.ylim(min_acc - y_margin, max_acc + y_margin)
        plt.xlim(1, max_epoch)
    else:
        plt.ylim(0, 1.05)
        plt.xlim(1, max_epoch)
    
    # 设置标签和标题
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    
    # 添加网格
    if show_grid:
        plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # 设置刻度
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 添加文本说明（可选）
    text_str = f"Total models: {len(all_data)}"
    plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {save_path}")
    plt.show()
    
    return all_data

def check_data_file(file_path):
    """检查数据文件是否存在且有效"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                print(f"文件为空: {file_path}")
                return False
            
            # 检查第一行是否包含数字
            first_line = lines[0].strip()
            if not re.search(r'\d', first_line):
                print(f"文件格式可能不正确: {file_path}")
                print(f"第一行内容: {first_line}")
                return False
    except Exception as e:
        print(f"读取文件失败: {e}")
        return False
    
    return True

if __name__ == '__main__':
    # 定义要绘制的文件路径列表
    # 方式1：直接提供文件路径列表
    accuracy_files = [
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/TSMFC1_CrowdViolence_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/TSMFC2_CrowdViolence_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/TSMFC3_CrowdViolence_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/TSMFC4_CrowdViolence_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
        '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/TSMFC5_CrowdViolence_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/validation_accuracy.txt',
    ]
    
    # 方式2：如果需要自定义标签，可以使用元组格式
    # accuracy_files = [
    #     ('/path/to/crowd_violence.txt', 'CrowdViolence'),
    #     ('/path/to/hockey_fights.txt', 'HockeyFights'),
    #     ('/path/to/rwf2000.txt', 'RWF-2000'),
    # ]
    
    # 检查文件是否存在
    valid_files = []
    for file_path in accuracy_files:
        if check_data_file(file_path):
            valid_files.append(file_path)
        else:
            print(f"跳过低效文件: {file_path}")
    
    if valid_files:
        print(f"\n找到 {len(valid_files)} 个有效文件，开始绘制对比曲线...")
        
        # 绘制多条曲线
        plot_multiple_accuracy_curves(
            valid_files,
            save_path='accuracy_curves_comparison.png',
            title='Model Accuracy Comparison on Different Datasets',
            figsize=(12, 7),
            auto_range=True,
            show_grid=True
        )
        
        # 也可以绘制不同的样式
        # plot_multiple_accuracy_curves(
        #     valid_files,
        #     save_path='accuracy_curves_comparison_detailed.png',
        #     title='Validation Accuracy Curves',
        #     figsize=(14, 8),
        #     auto_range=False,  # 使用固定范围 0-1.05
        #     show_grid=True
        # )
    else:
        print("没有找到有效的准确率文件，请检查文件路径和格式")
        
        # 如果文件不存在，可以尝试搜索
        search_paths = [
            '/mnt/data-windows/jjz/DualCascade TSF_MobileNetV2/log/'
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                print(f"\n在 {search_path} 中搜索 validation_loss.txt 文件...")
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file == 'validation_loss.txt':
                            print(f"找到: {os.path.join(root, file)}")