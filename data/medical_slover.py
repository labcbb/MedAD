import os
import json


class MedicalSolver(object):
    CLSNAMES = ["ISIC18"]  # 你的类别目录名

    def __init__(self, root="./data/medical"):
        self.root = root
        self.meta_path = f"{root}/meta.json"
        # 支持的图像格式
        self.supported_image_exts = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]
        # 支持的掩膜格式
        self.supported_mask_exts = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]

    def get_image_files(self, directory):
        """获取目录中所有支持的图像文件"""
        if not os.path.exists(directory):
            return []
        
        all_files = os.listdir(directory)
        image_files = []
        for file in all_files:
            if any(file.endswith(ext) for ext in self.supported_image_exts):
                image_files.append(file)
        return sorted(image_files)
        
    def get_mask_files(self, directory):
        """获取目录中所有支持的掩膜文件"""
        if not os.path.exists(directory):
            return []
        
        all_files = os.listdir(directory)
        mask_files = []
        for file in all_files:
            if any(file.endswith(ext) for ext in self.supported_mask_exts):
                mask_files.append(file)
        return sorted(mask_files)

    def find_corresponding_mask(self, img_name, mask_names):
        """根据图像文件名找到对应的掩膜文件"""
        # 移除图像文件的扩展名
        img_base = img_name
        for ext in self.supported_image_exts:
            if img_base.endswith(ext):
                img_base = img_base[:-len(ext)]
                break
        
        # 尝试多种掩膜命名模式
        possible_mask_names = [
            f"{img_base}_mask.png",
            f"{img_base}_mask.jpg",
            f"{img_base}_mask.jpeg",
            f"{img_base}.png",
            f"{img_base}.jpg", 
            f"{img_base}.jpeg",
            img_base + ".png",
            img_base + ".jpg",
            img_base + ".jpeg"
        ]
        
        for mask_name in possible_mask_names:
            if mask_name in mask_names:
                return mask_name
        
        return None

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f"{self.root}/{cls_name}"
            for phase in ["train", "test"]:
                cls_info = []
                # 跳过隐藏文件夹（比如 .DS_Store）
                species = [s for s in os.listdir(f"{cls_dir}/{phase}") if not s.startswith(".")]
                for specie in species:
                    is_abnormal = True if specie != "good" else False
                    img_dir = f"{cls_dir}/{phase}/{specie}"
                    
                    # 获取所有支持的图像文件
                    img_names = self.get_image_files(img_dir)
                    if not img_names:
                        print(f"⚠️ 警告: 在 {img_dir} 中没有找到支持的图像文件")
                        continue

                    # 检查是否有ground_truth目录（可选）
                    has_ground_truth = False
                    mask_names = []
                    if is_abnormal:
                        mask_dir = f"{cls_dir}/ground_truth/{specie}"
                        if os.path.exists(mask_dir):
                            mask_names = self.get_mask_files(mask_dir)
                            has_ground_truth = len(mask_names) > 0
                            if not has_ground_truth:
                                print(f"ℹ️ 信息: {mask_dir} 存在但为空，将作为分类数据集处理")
                        else:
                            print(f"ℹ️ 信息: {mask_dir} 不存在，将作为分类数据集处理")

                    for idx, img_name in enumerate(img_names):
                        mask_path = ""
                        
                        # 只有在有ground_truth且为异常样本时才尝试匹配掩膜
                        if is_abnormal and has_ground_truth:
                            mask_name = self.find_corresponding_mask(img_name, mask_names)
                            if mask_name:
                                mask_path = f"{cls_name}/ground_truth/{specie}/{mask_name}"
                            else:
                                print(f"⚠️ 警告: {img_name} 没有找到对应的掩膜文件")

                        info_img = dict(
                            img_path=f"{cls_name}/{phase}/{specie}/{img_name}",
                            mask_path=mask_path,
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)

                info[phase][cls_name] = cls_info

        with open(self.meta_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(info, indent=4, ensure_ascii=False) + "\n")

        print(f"✅ meta.json 已生成: {self.meta_path}")
        
        # 统计信息
        total_train = sum(len(info["train"][cls]) for cls in info["train"])
        total_test = sum(len(info["test"][cls]) for cls in info["test"])
        print(f"📊 统计信息:")
        print(f"   - 训练集: {total_train} 张图像")
        print(f"   - 测试集: {total_test} 张图像")
        for cls_name in info["train"]:
            train_count = len(info["train"][cls_name])
            test_count = len(info["test"][cls_name])
            train_abnormal = sum(1 for item in info["train"][cls_name] if item["anomaly"] == 1)
            test_abnormal = sum(1 for item in info["test"][cls_name] if item["anomaly"] == 1)
            print(f"   - {cls_name}: 训练{train_count}张(异常{train_abnormal}张), 测试{test_count}张(异常{test_abnormal}张)")


if __name__ == "__main__":
    runner = MedicalSolver(root="/share/home/yangLab/shaozhixiong/MedAD/data/medical")
    runner.run()
