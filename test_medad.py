import os
# 设置PyTorch内存分配优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import logging
import numpy as np
import torch
import torchvision
import threading
import torchvision.transforms as transforms
from tabulate import tabulate
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import math
from PIL import Image
from prefetch_generator import BackgroundGenerator
from MedAD import MedAD
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import time  # 用于性能统计
import gc  # 垃圾回收

from datasets.mvtec import MVTecDataset
from datasets.visa import VisaDataset
from datasets.mvtec_loco import MVTecLocoDataset
from datasets.brainmri import BrainMRIDataset
from datasets.his import HISDataset
from datasets.resc import RESCDataset
from datasets.liverct import LiverCTDataset
from datasets.chestxray import ChestXrayDataset
from datasets.oct17 import OCT17Dataset


class DataLoaderX(torch.utils.data.DataLoader):
    """自定义数据加载器，使用后台生成器优化数据加载性能"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def safe_auroc(y_true, y_score):
    """安全的AUROC计算，处理单一标签情况"""
    try:
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        
        if len(y_true) == 0 or len(y_score) == 0:
            return np.nan
        if len(y_true) != len(y_score):
            return np.nan
        if len(np.unique(y_true)) < 2:
            return np.nan
            
        y_true = y_true.astype(float)
        y_score = y_score.astype(float)
        
        return roc_auc_score(y_true, y_score)
    except Exception as e:
        print(f"AUROC计算错误: {str(e)}")
        return np.nan


def save_visualization(image_pil, gt_mask, anomaly_map, anomaly_score, save_dir, filename, image_size=224):
    """保存可视化结果"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(image_pil, list):
            image_pil = image_pil[0]

        if isinstance(image_pil, torch.Tensor):
            if image_pil.dim() == 4:
                image_tensor = image_pil.squeeze(0)
            else:
                image_tensor = image_pil
            if image_tensor.shape[0] == 3:
                image_tensor = image_tensor.permute(1, 2, 0)
            image_np = image_tensor.detach().cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            image_pil = Image.fromarray(image_np)

        image_resized = image_pil.resize((image_size, image_size))
        image_np = np.array(image_resized)

        gt_mask_np = gt_mask.squeeze().numpy()
        gt_mask_resized = cv2.resize(gt_mask_np, (image_size, image_size)) if gt_mask_np.shape != (image_size, image_size) else gt_mask_np

        if len(anomaly_map.shape) > 2:
            anomaly_map = anomaly_map.squeeze()
        if not isinstance(anomaly_map, np.ndarray):
            anomaly_map = np.array(anomaly_map)
        if len(anomaly_map.shape) != 2:
            anomaly_map = np.zeros((image_size, image_size))
        anomaly_map_resized = cv2.resize(anomaly_map.astype(np.float32), (image_size, image_size)) if anomaly_map.shape != (image_size, image_size) else anomaly_map
        anomaly_map_norm = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.max() - anomaly_map_resized.min() + 1e-8)

        heatmap = cm.jet(anomaly_map_norm)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(image_np)
        plt.title('Original Image', fontsize=12)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(gt_mask_resized, cmap='gray')
        plt.title('Ground Truth', fontsize=12)
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(heatmap)
        plt.title(f'Anomaly Heatmap\nScore: {anomaly_score:.3f}', fontsize=12)
        plt.axis('off')

        plt.subplot(1, 4, 4)
        overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
        plt.imshow(overlay)
        plt.title('Overlay View', fontsize=12)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{filename}_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"⚠️  保存可视化文件时出错: {filename} - {str(e)}")
        return False


def cal_score(obj):
    """计算指定对象的评估指标"""
    table = []
    gt_px, pr_px, gt_sp, pr_sp = [], [], [], []

    table.append(obj)
    for idxes in range(len(results["cls_names"])):
        if results["cls_names"][idxes] == obj:
            # 参考代码的简单处理方式
            gt_px.append(results["imgs_masks"][idxes].squeeze(1).numpy())
            pr_px.append(results["anomaly_maps"][idxes])
            gt_sp.append(results["gt_sp"][idxes])
            pr_sp.append(results["pr_sp"][idxes])
    
    # 直接转换为numpy数组（参考代码方式）
    gt_px = np.array(gt_px)
    gt_sp = np.array(gt_sp)
    pr_px = np.array(pr_px)
    pr_sp = np.array(pr_sp)

    auroc_sp = safe_auroc(gt_sp, pr_sp)
    auroc_px = safe_auroc(gt_px.ravel(), pr_px.ravel())  # 参考代码的简单方式

    table.append(str(np.round(auroc_sp * 100, decimals=1)) if not np.isnan(auroc_sp) else "nan")
    table.append(str(np.round(auroc_px * 100, decimals=1)) if not np.isnan(auroc_px) else "nan")

    table_ls.append(table)
    if not np.isnan(auroc_sp):
        auroc_sp_ls.append(auroc_sp)
    if not np.isnan(auroc_px):
        auroc_px_ls.append(auroc_px)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test", add_help=True)
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--k_shot", type=int, default=1, help="k-shot")
    parser.add_argument("--dataset", type=str, default="mvtec", help="train dataset name")
    parser.add_argument("--data_path", type=str, default="./data/mvtec", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default=f"./results/", help="path to save results")
    parser.add_argument("--round", type=int, default=0, help="round")
    parser.add_argument("--class_name", type=str, default="None", help="device")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--save_vis", action="store_true", help="是否保存可视化结果")
    parser.add_argument("--vis_num", type=int, default=500, help="每个类别保存的可视化图像数量")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32", 
                       choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336"],
                       help="CLIP模型选择：ViT-B-32(patch32), ViT-B-16(patch16), ViT-L-14(patch14), ViT-L-14-336(patch14,支持336尺寸)")
    
    # 消融实验参数
    parser.add_argument("--use_clip", action="store_true", default=True, help="是否使用CLIP特征提取器")
    parser.add_argument("--use_dino", action="store_true", default=True, help="是否使用DINOv2特征提取器")
    parser.add_argument("--use_text", action="store_true", default=True, help="是否使用文本特征")
    parser.add_argument("--ablation_mode", type=str, default="all", 
                       choices=["all", "clip_only", "dino_only", "text_only", "clip_dino", "clip_text", "dino_text", "none"],
                       help="消融实验模式：all(全部), clip_only(仅CLIP), dino_only(仅DINO), text_only(仅文本), clip_dino(CLIP+DINO), clip_text(CLIP+文本), dino_text(DINO+文本), none(无特征)")
    
    # 跨数据集泛化测试参数
    parser.add_argument("--cross_dataset", action="store_true", help="启用跨数据集泛化测试")
    parser.add_argument("--source_dataset", type=str, default="", help="源数据集名称（用于提供正常样本）")
    parser.add_argument("--source_class", type=str, default="", help="源数据集类别名称")
    parser.add_argument("--source_data_path", type=str, default="", help="源数据集路径")
    parser.add_argument("--target_datasets", type=str, nargs="+", default=[], help="目标数据集列表，如：medical visa mvtec_loco")
    parser.add_argument("--target_data_paths", type=str, nargs="+", default=[], help="目标数据集路径列表")
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir = args.data_path
    device = args.device
    k_shot = args.k_shot
    image_size = args.image_size
    
    # 处理消融实验模式
    if args.ablation_mode != "all":
        if args.ablation_mode == "clip_only":
            use_clip, use_dino, use_text = True, False, False
        elif args.ablation_mode == "dino_only":
            use_clip, use_dino, use_text = False, True, False
        elif args.ablation_mode == "text_only":
            use_clip, use_dino, use_text = False, False, True
        elif args.ablation_mode == "clip_dino":
            use_clip, use_dino, use_text = True, True, False
        elif args.ablation_mode == "clip_text":
            use_clip, use_dino, use_text = True, False, True
        elif args.ablation_mode == "dino_text":
            use_clip, use_dino, use_text = False, True, True
        elif args.ablation_mode == "none":
            use_clip, use_dino, use_text = False, False, False
    else:
        use_clip = args.use_clip
        use_dino = args.use_dino
        use_text = args.use_text
    
    # 根据消融实验模式和跨数据集模式调整保存路径
    ablation_suffix = f"_{args.ablation_mode}" if args.ablation_mode != "all" else ""
    cross_dataset_suffix = f"_cross_{args.source_dataset}_{args.source_class}" if args.cross_dataset else ""
    clip_suffix = f"_{args.clip_model.replace('-', '_')}" if args.clip_model != "ViT-B-32" else ""
    save_path = args.save_path + "/" + dataset_name + ablation_suffix + cross_dataset_suffix + clip_suffix + "/"
    os.makedirs(save_path, exist_ok=True)
    txt_path = os.path.join(save_path, "log.txt")

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger("test")
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # 设备安全检查
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("⚠️  CUDA不可用，自动切换到CPU")
    
    MedAD_model = MedAD(image_size=image_size, use_clip=use_clip, use_dino=use_dino, use_text=use_text, clip_model_name=args.clip_model).to(device)
    MedAD_model.eval()  # 设置为评估模式
    
    # 尝试为CLIP与DINO编码器注入前向计时包装器（无需改动模型源码）
    def _wrap_module_forward_for_timing(parent, attr_candidates, save_attr_name):
        for name in attr_candidates:
            if hasattr(parent, name):
                mod = getattr(parent, name)
                if callable(getattr(mod, 'forward', None)):
                    orig_forward = mod.forward
                    def timed_forward(*f_args, **f_kwargs):
                        t0 = time.time()
                        # CUDA同步确保计时精确
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        except Exception:
                            pass
                        out = orig_forward(*f_args, **f_kwargs)
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        except Exception:
                            pass
                        t1 = time.time()
                        setattr(MedAD_model, save_attr_name, float(t1 - t0))
                        return out
                    mod.forward = timed_forward
                    return True
        return False
    
    # 注入计时：尽最大可能匹配到子编码器属性名
    _wrap_module_forward_for_timing(MedAD_model, ["clip_net", "clip", "clip_model"], "_clip_time")
    _wrap_module_forward_for_timing(MedAD_model, ["dino_net", "dino", "dino_model"], "_dino_time")
    
    # 记录消融实验配置
    logger.info(f"消融实验配置: CLIP={use_clip}, DINO={use_dino}, Text={use_text}")
    logger.info(f"消融实验模式: {args.ablation_mode}")
    logger.info(f"CLIP模型: {args.clip_model}, Patch Size: {MedAD_model.clip_patch_size if hasattr(MedAD_model, 'clip_patch_size') else 'N/A'}")
    
    # 记录跨数据集配置
    if args.cross_dataset:
        logger.info(f"🔄 跨数据集泛化测试模式")
        logger.info(f"源数据集: {args.source_dataset}, 源类别: {args.source_class}")
        logger.info(f"目标数据集: {dataset_name}")
        logger.info(f"保存路径: {save_path}")

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    # dataset初始化（保持原路径）
    if dataset_name == "medical":
        test_data = MVTecDataset(root=dataset_dir, transform=transform, target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "visa":
        test_data = VisaDataset(root=dataset_dir, transform=transform, target_transform=transform, mode="test")
    elif dataset_name == "mvtec_loco":
        test_data = MVTecLocoDataset(root=dataset_dir, transform=transform, target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "brainmri":
        test_data = BrainMRIDataset(root="./data/BrainMRI", transform=transform, target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "his":
        test_data = HISDataset(root="./data/HIS", transform=transform, target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "resc":
        test_data = RESCDataset(root="./data/RESC", transform=transform, target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "chestxray":
        test_data = ChestXrayDataset(root="./data/chestxray", transform=transform, target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "liverct":
        test_data = LiverCTDataset(root="./data/liverct", transform=transform, target_transform=transform, aug_rate=-1, mode="test")
    elif dataset_name == "oct17":
        test_data = OCT17Dataset(root="./data/oct17", transform=transform, target_transform=transform, aug_rate=-1, mode="test")
    else:
        raise NotImplementedError

    dataloader = DataLoaderX(test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    results = {"imgs_masks": [], "anomaly_maps": [], "gt_sp": [], "pr_sp": [], "cls_names": []}
    vis_count_per_class = {}
    cls_last = None  # 跟踪上一个类别，用于setup

    # 用于setup的图像变换
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor()
    ])

    # 跨数据集泛化测试：加载源数据集的正常样本
    source_normal_images = None
    source_normal_paths = []
    if args.cross_dataset:
        if not args.source_dataset or not args.source_class or not args.source_data_path:
            raise ValueError("跨数据集测试需要指定 --source_dataset, --source_class, --source_data_path")
        
        logger.info(f"🔄 跨数据集泛化测试模式")
        logger.info(f"源数据集: {args.source_dataset}, 类别: {args.source_class}, 路径: {args.source_data_path}")
        
        # 构建源数据集正常样本路径
        if args.source_dataset == "medical":
            source_normal_dir = os.path.join(args.source_data_path, args.source_class.replace(" ", "_"), "train", "good")
        elif args.source_dataset in ["his", "oct17", "chestxray", "brainmri", "liverct", "resc"]:
            source_normal_dir = os.path.join("./data", args.source_class.replace(" ", "_"), "train", "good")
        else:
            source_normal_dir = os.path.join(args.source_data_path, args.source_class.replace(" ", "_"), "train", "good")
        
        if os.path.exists(source_normal_dir):
            files = [f for f in sorted(os.listdir(source_normal_dir)) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')) 
                    and not f.startswith('.')][:k_shot]
            source_normal_paths = [os.path.join(source_normal_dir, file) for file in files]
            logger.info(f"找到 {len(source_normal_paths)} 个源数据集正常样本")
        else:
            raise FileNotFoundError(f"源数据集正常样本目录不存在: {source_normal_dir}")
        
        # 加载源数据集正常样本
        source_images_list = []
        for path in source_normal_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = image_transform(img).unsqueeze(0)
                source_images_list.append(img_tensor)
            except Exception as e:
                logger.warning(f"无法加载源数据集图像 {path}: {str(e)}")
                continue
        
        if not source_images_list:
            raise ValueError("源数据集没有有效的正常样本图像")
        
        source_normal_images = torch.cat(source_images_list, dim=0).to(device)
        logger.info(f"✅ 源数据集正常样本加载完成，共 {len(source_images_list)} 个样本")

    start_time = time.time()
    inference_times = []  # 记录每张图片的推理时间
    clip_times = []       # 记录每张图片的CLIP特征提取时间（若模型提供）
    dino_times = []       # 记录每张图片的DINOv2特征提取时间（若模型提供）

    # -------------- 把 no_grad 放到整个循环外 --------------
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Processing")):
            # 数据集返回字典格式，需要正确解包
            image = data['img']
            gt_mask = data['img_mask'] 
            cls_name = data['cls_name']
            image_path = data['img_path']  # 获取图像路径
            if isinstance(cls_name, list):
                cls_name = cls_name[0]
            if isinstance(image_path, list):
                image_path = image_path[0]
            
            if cls_name not in vis_count_per_class:
                vis_count_per_class[cls_name] = 0

            # 如果是新类别，需要重新setup模型
            if cls_name != cls_last:
                logger.info(f"为类别 '{cls_name}' 设置模型...")
                
                # 跨数据集模式：使用源数据集的正常样本
                if args.cross_dataset and source_normal_images is not None:
                    logger.info(f"🔄 使用源数据集正常样本进行跨数据集测试")
                    setup_data = {
                        "few_shot_samples": source_normal_images,
                        "dataset_category": cls_name.replace(" ", "_"),
                        "image_path": source_normal_paths,
                    }
                    MedAD_model.setup(setup_data)
                    cls_last = cls_name
                    logger.info(f"类别 '{cls_name}' 跨数据集设置完成，使用源数据集 {args.source_class} 的 {len(source_normal_paths)} 个正常样本")
                    # 打印该类别的gate类型
                    try:
                        logger.info(f"类别 '{cls_name}' 的gate类型: {MedAD_model.gate.name}")
                    except Exception:
                        logger.info(f"类别 '{cls_name}' 的gate类型: 未知")
                else:
                    # 常规模式：使用当前数据集的正常样本
                    # 根据数据集构建正常样本路径
                    if dataset_name == "medical":
                        normal_dir = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train", "good")
                        if os.path.exists(normal_dir):
                            files = [f for f in sorted(os.listdir(normal_dir)) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')) 
                                    and not f.startswith('.')][:k_shot]
                            normal_image_paths = [os.path.join(normal_dir, file) for file in files]
                        else:
                            # 如果找不到正常样本目录，使用默认路径
                            normal_image_paths = [
                                os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train", "good", f"{str(j).zfill(3)}.png")
                                for j in range(args.round, args.round + k_shot)
                            ]
                    elif dataset_name in ["his", "oct17", "chestxray", "brainmri", "liverct", "resc"]:
                        normal_dir = os.path.join("./data", cls_name.replace(" ", "_"), "train", "good")
                        if os.path.exists(normal_dir):
                            files = [f for f in sorted(os.listdir(normal_dir)) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')) 
                                    and not f.startswith('.')][:k_shot]
                            normal_image_paths = [os.path.join(normal_dir, file) for file in files]
                        else:
                            logger.warning(f"找不到正常样本目录: {normal_dir}")
                            continue
                    else:
                        # 通用路径构建
                        normal_dir = os.path.join(dataset_dir, cls_name.replace(" ", "_"), "train", "good")
                        if os.path.exists(normal_dir):
                            files = [f for f in sorted(os.listdir(normal_dir)) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')) 
                                    and not f.startswith('.')][:k_shot]
                            normal_image_paths = [os.path.join(normal_dir, file) for file in files]
                        else:
                            logger.warning(f"找不到正常样本目录: {normal_dir}，跳过类别 {cls_name}")
                            continue
                    
                    # Zero-shot 兼容：当 k_shot==0 时不强制需要正常样本
                    if k_shot == 0:
                        setup_data = {
                            "few_shot_samples": None,
                            "dataset_category": cls_name.replace(" ", "_"),
                            "image_path": [],
                        }
                        MedAD_model.setup(setup_data)
                        cls_last = cls_name
                        logger.info(f"类别 '{cls_name}' 设置完成（zero-shot 模式）")
                        # 打印该类别的gate类型
                        try:
                            logger.info(f"类别 '{cls_name}' 的gate类型: {MedAD_model.gate.name}")
                        except Exception:
                            logger.info(f"类别 '{cls_name}' 的gate类型: 未知")
                    else:
                        # 检查正常样本文件是否存在
                        valid_paths = []
                        for path in normal_image_paths:
                            if os.path.exists(path):
                                valid_paths.append(path)
                            else:
                                logger.warning(f"正常样本文件不存在: {path}")
                        if not valid_paths:
                            logger.error(f"类别 '{cls_name}' 没有找到有效的正常样本，跳过")
                            continue
                        normal_image_paths = valid_paths[:k_shot]  # 限制k_shot数量

                        try:
                            # 加载正常样本图像
                            normal_images_list = []
                            for path in normal_image_paths:
                                try:
                                    img = Image.open(path).convert("RGB")
                                    img_tensor = image_transform(img).unsqueeze(0)
                                    normal_images_list.append(img_tensor)
                                except Exception as e:
                                    logger.warning(f"无法加载图像 {path}: {str(e)}")
                                    continue
                            if not normal_images_list:
                                logger.error(f"类别 '{cls_name}' 没有有效的正常样本图像")
                                continue
                            normal_images = torch.cat(normal_images_list, dim=0).to(device)

                            # 构建setup数据
                            setup_data = {
                                "few_shot_samples": normal_images,
                                "dataset_category": cls_name.replace(" ", "_"),
                                "image_path": normal_image_paths,
                            }
                            # 执行setup
                            MedAD_model.setup(setup_data)
                            cls_last = cls_name
                            logger.info(f"类别 '{cls_name}' 设置完成，使用 {len(normal_image_paths)} 个正常样本")
                            # 打印该类别的gate类型
                            try:
                                logger.info(f"类别 '{cls_name}' 的gate类型: {MedAD_model.gate.name}")
                            except Exception:
                                logger.info(f"类别 '{cls_name}' 的gate类型: 未知")
                        except Exception as e:
                            logger.error(f"为类别 '{cls_name}' 执行setup时出错: {str(e)}")
                            continue

            # 模型推理 - 记录推理时间
            image = image.to(device)
            try:
                # 记录推理开始时间
                inference_start = time.time()
                result = MedAD_model(image, image_path)
                # 记录推理结束时间
                inference_end = time.time()
                inference_time = inference_end - inference_start
                inference_times.append(inference_time)
                # 若模型在forward中记录了组件耗时，则读取
                try:
                    if hasattr(MedAD_model, "_clip_time") and isinstance(MedAD_model._clip_time, (int, float)):
                        clip_times.append(float(MedAD_model._clip_time))
                    if hasattr(MedAD_model, "_dino_time") and isinstance(MedAD_model._dino_time, (int, float)):
                        dino_times.append(float(MedAD_model._dino_time))
                except Exception:
                    pass
                
                # 从返回的字典中提取结果
                anomaly_map = result["pred_mask"].squeeze().detach().cpu().numpy()
                anomaly_score = result["pred_score"]
                
                # 立即清理推理结果，释放GPU显存
                del result
                
                # 确保anomaly_score为标量
                if isinstance(anomaly_score, torch.Tensor):
                    anomaly_score = float(anomaly_score.detach().cpu().item() if anomaly_score.numel() == 1 else anomaly_score.mean().detach().cpu().item())
                    
            except Exception as e:
                logger.error(f"推理第 {i} 个样本时出错: {str(e)}")
                continue

            # 从数据集中获取真实的异常标签
            anomaly_label = data.get('anomaly', 0)  # 获取异常标签，默认为0（正常）
            if isinstance(anomaly_label, torch.Tensor):
                anomaly_label = float(anomaly_label.item())
            elif isinstance(anomaly_label, list):
                anomaly_label = float(anomaly_label[0])
            else:
                anomaly_label = float(anomaly_label)

            # 对gt_mask进行二值化处理（关键！）
            gt_mask_processed = gt_mask.clone()
            gt_mask_processed[gt_mask_processed > 0.5] = 1
            gt_mask_processed[gt_mask_processed <= 0.5] = 0

            # 保存结果（只保存必要的数据，避免累积大量张量）
            # 注意：不保存原始图像张量，只保存标量结果
            results["imgs_masks"].append(gt_mask_processed.cpu())  # 转移到CPU
            results["anomaly_maps"].append(anomaly_map)  # 已经是numpy数组
            results["gt_sp"].append(anomaly_label)  # 使用真实的异常标签
            results["pr_sp"].append(anomaly_score)
            results["cls_names"].append(cls_name)
            
            # 立即清理GPU上的张量
            del image, gt_mask_processed

            # 可视化保存
            if args.save_vis and vis_count_per_class[cls_name] < args.vis_num:
                # 以数据集根目录为基准，保持原有层级结构，并区分正常/异常
                try:
                    rel_path = os.path.relpath(image_path, dataset_dir)
                except Exception:
                    rel_path = os.path.basename(image_path)
                sub_split = "defect" if anomaly_label == 1 else "good"
                save_dir = os.path.join(save_path, "visualizations_structured", sub_split, os.path.dirname(rel_path))
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.splitext(os.path.basename(image_path))[0]
                # 重新加载图像用于可视化（避免使用已删除的GPU张量）
                try:
                    vis_image = Image.open(image_path).convert("RGB")
                    vis_image_tensor = transform(vis_image).unsqueeze(0)
                    saved = save_visualization(vis_image_tensor, gt_mask, anomaly_map, anomaly_score, save_dir, filename, image_size=image_size)
                    if saved:
                        vis_count_per_class[cls_name] += 1
                    del vis_image_tensor
                except Exception as e:
                    logger.warning(f"可视化保存失败: {str(e)}")
            
            # 定期清理GPU内存
            if i % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    total_time = time.time() - start_time
    
    # 计算推理时间统计
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        total_inference_time = np.sum(inference_times)
        fps = len(inference_times) / total_inference_time if total_inference_time > 0 else 0
        
        logger.info(f"✅ 处理完成，总耗时: {total_time:.1f} 秒")
        logger.info(f"📊 推理时间统计:")
        logger.info(f"  - 平均推理时间: {avg_inference_time*1000:.2f} ms")
        logger.info(f"  - 标准差: {std_inference_time*1000:.2f} ms")
        logger.info(f"  - 最快推理: {min_inference_time*1000:.2f} ms")
        logger.info(f"  - 最慢推理: {max_inference_time*1000:.2f} ms")
        logger.info(f"  - 总推理时间: {total_inference_time:.2f} s")
        logger.info(f"  - 推理FPS: {fps:.2f}")
        logger.info(f"  - 处理图片数: {len(inference_times)}")

        # 组件级时间统计：CLIP & DINOv2（若已收集）
        if clip_times:
            logger.info(f"🔍 CLIP特征提取时间:")
            logger.info(f"  - 平均: {np.mean(clip_times)*1000:.2f} ms")
            logger.info(f"  - 标准差: {np.std(clip_times)*1000:.2f} ms")
        if dino_times:
            logger.info(f"🔍 DINOv2特征提取时间:")
            logger.info(f"  - 平均: {np.mean(dino_times)*1000:.2f} ms")
            logger.info(f"  - 标准差: {np.std(dino_times)*1000:.2f} ms")
    else:
        logger.info(f"✅ 处理完成，总耗时: {total_time:.1f} 秒")
        logger.info(f"⚠️  没有记录到推理时间数据")

    # 计算AUROC
    table_ls, auroc_sp_ls, auroc_px_ls = [], [], []
    cls_unique = list(set(results["cls_names"]))
    threads = []
    for obj in cls_unique:
        t = threading.Thread(target=cal_score, args=(obj,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print(tabulate(table_ls, headers=["objects", "auroc_sp", "auroc_px"], tablefmt="github"))
    avg_auroc_sp = np.nanmean(auroc_sp_ls) * 100 if auroc_sp_ls else 0
    avg_auroc_px = np.nanmean(auroc_px_ls) * 100 if auroc_px_ls else 0
    logger.info(f"平均 AUROC_sp: {avg_auroc_sp:.2f}%, 平均 AUROC_px: {avg_auroc_px:.2f}%")
    
    # 跨数据集结果统计
    if args.cross_dataset:
        logger.info(f"🔄 跨数据集泛化测试结果总结:")
        logger.info(f"源数据集: {args.source_dataset} - {args.source_class}")
        logger.info(f"目标数据集: {dataset_name}")
        logger.info(f"测试类别数: {len(cls_unique)}")
        logger.info(f"平均图像级AUROC: {avg_auroc_sp:.2f}%")
        logger.info(f"平均像素级AUROC: {avg_auroc_px:.2f}%")
        if inference_times:
            logger.info(f"平均推理时间: {np.mean(inference_times)*1000:.2f} ms")
            logger.info(f"推理FPS: {len(inference_times) / np.sum(inference_times):.2f}")
        
        # 保存跨数据集结果摘要
        cross_summary_path = os.path.join(save_path, "cross_dataset_summary.txt")
        with open(cross_summary_path, "w", encoding="utf-8") as f:
            f.write(f"跨数据集泛化测试结果摘要\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"源数据集: {args.source_dataset}\n")
            f.write(f"源类别: {args.source_class}\n")
            f.write(f"目标数据集: {dataset_name}\n")
            f.write(f"测试类别数: {len(cls_unique)}\n")
            f.write(f"平均图像级AUROC: {avg_auroc_sp:.2f}%\n")
            f.write(f"平均像素级AUROC: {avg_auroc_px:.2f}%\n")
            if inference_times:
                f.write(f"平均推理时间: {np.mean(inference_times)*1000:.2f} ms\n")
                f.write(f"推理FPS: {len(inference_times) / np.sum(inference_times):.2f}\n")
                f.write(f"处理图片数: {len(inference_times)}\n")
            f.write(f"\n详细结果:\n")
            f.write(tabulate(table_ls, headers=["objects", "auroc_sp", "auroc_px"], tablefmt="github"))
        logger.info(f"跨数据集结果摘要已保存到: {cross_summary_path}")
