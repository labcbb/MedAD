from models.component_segmentaion import grounding_segmentation
import os
import glob
import yaml
import argparse
from pathlib import Path

# 支持的图像格式
SUPPORTED_IMAGE_EXTENSIONS = [
    '*.png', '*.PNG',
    '*.jpg', '*.JPG', '*.jpeg', '*.JPEG',
    '*.bmp', '*.BMP',
    '*.tiff', '*.TIFF', '*.tif', '*.TIF',
    '*.webp', '*.WEBP'
]

def read_config(config_path):
    """读取YAML配置文件"""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def find_images_in_directory(directory, extensions=None):
    """在指定目录中查找所有支持的图像文件"""
    if extensions is None:
        extensions = SUPPORTED_IMAGE_EXTENSIONS
    
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(directory, "**", ext)
        image_paths.extend(glob.glob(pattern, recursive=True))
    
    return sorted(image_paths)

def process_dataset(data_root, mask_root, categories, config_root="./configs/class_histogram", 
                   splits=['train', 'test'], extensions=None):
    """
    处理数据集的通用函数
    
    Args:
        data_root: 数据根目录
        mask_root: 掩码输出根目录
        categories: 类别列表
        config_root: 配置文件根目录
        splits: 数据分割列表 ['train', 'test']
        extensions: 图像扩展名列表，None表示使用默认支持的所有格式
    """
    if extensions is None:
        extensions = SUPPORTED_IMAGE_EXTENSIONS
    
    print(f"🔍 开始处理数据集: {data_root}")
    print(f"📁 输出目录: {mask_root}")
    print(f"📋 类别: {categories}")
    print(f"🖼️  支持的图像格式: {extensions}")
    
    for category in categories:
        print(f"\n📂 处理类别: {category}")
        
        # 检查配置文件是否存在
        config_path = os.path.join(config_root, f"{category}.yaml")
        if not os.path.exists(config_path):
            print(f"⚠️  配置文件不存在: {config_path}")
            continue
        
        try:
            config = read_config(config_path)
        except Exception as e:
            print(f"❌ 读取配置文件失败: {e}")
            continue
        
        # 为每个数据分割处理
        for split in splits:
            print(f"  📊 处理 {split} 数据...")
            
            # 构建数据路径
            data_path = os.path.join(data_root, category, split)
            if not os.path.exists(data_path):
                print(f"    ⚠️  数据目录不存在: {data_path}")
                continue
            
            # 查找图像文件
            image_paths = find_images_in_directory(data_path, extensions)
            if not image_paths:
                print(f"    ⚠️  在 {data_path} 中未找到图像文件")
                continue
            
            print(f"    🖼️  找到 {len(image_paths)} 张图像")
            
            # 创建输出目录
            output_dir = os.path.join(mask_root, category)
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                # 执行分割
                grounding_segmentation(
                    image_paths, output_dir, config["grounding_config"]
                )
                print(f"    ✅ {split} 数据分割完成")
            except Exception as e:
                print(f"    ❌ {split} 数据分割失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='通用图像分割处理脚本')
    parser.add_argument('--data_root', type=str, default='./data/medical',
                       help='数据根目录')
    parser.add_argument('--mask_root', type=str, default='./masks/medical',
                       help='掩码输出根目录')
    parser.add_argument('--config_root', type=str, default='./configs/class_histogram',
                       help='配置文件根目录')
    parser.add_argument('--categories', nargs='+', default=['ISIC18'],
                       help='要处理的类别列表')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                       help='数据分割列表')
    parser.add_argument('--extensions', nargs='+', default=None,
                       help='图像扩展名列表，如 png jpg jpeg')
    parser.add_argument('--auto_detect_categories', action='store_true',
                       help='自动检测数据根目录下的所有类别')
    
    args = parser.parse_args()
    
    # 自动检测类别
    if args.auto_detect_categories:
        if os.path.exists(args.data_root):
            categories = [d for d in os.listdir(args.data_root) 
                         if os.path.isdir(os.path.join(args.data_root, d))]
            print(f"🔍 自动检测到类别: {categories}")
        else:
            print(f"❌ 数据根目录不存在: {args.data_root}")
            return
    else:
        categories = args.categories
    
    # 处理扩展名
    extensions = args.extensions
    if extensions:
        # 添加通配符
        extensions = [f"*.{ext}" if not ext.startswith('*') else ext for ext in extensions]
        extensions.extend([ext.upper() for ext in extensions if ext.islower()])
    
    # 执行处理
    process_dataset(
        data_root=args.data_root,
        mask_root=args.mask_root,
        categories=categories,
        config_root=args.config_root,
        splits=args.splits,
        extensions=extensions
    )
    
    print("\n🎉 所有处理完成！")

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认配置
    import sys
    if len(sys.argv) == 1:
        # 默认处理medical数据集
        print("🚀 使用默认配置处理medical数据集...")
        process_dataset(
            data_root="./data/medical",
            mask_root="./masks/medical", 
            categories=["APTOS19"],
            config_root="./configs/class_histogram",
            splits=['train', 'test']
        )
    else:
        main()
