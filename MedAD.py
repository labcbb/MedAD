import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import cv2
from modules import DinoFeaturizer, LinearLayer

from models.clip_prompt import encode_text_with_prompt_ensemble
from models.clip_prompt import get_prompt_sentences_for_object
from utils.filter_algorithm import filter_bg_noise
from utils.sampler import GreedyCoresetSampler
from utils.crf import dense_crf
import models.clip as open_clip
import os

from models.component_feature_extractor import ComponentFeatureExtractor
from models.component_segmentaion import (
    split_masks_from_one_mask,
    split_masks_from_one_mask_torch,
    split_masks_from_one_mask_with_bg
)

from matplotlib import pyplot as plt
from PIL import Image
from enum import Enum


class object_type(Enum):
    TEXTURE = 0
    SINGLE = 1
    MULTI = 2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
i_m = np.array(IMAGENET_MEAN)
i_m = i_m[:, None, None]
i_std = np.array(IMAGENET_STD)
i_std = i_std[:, None, None]


def get_heatmaps(img, query_feature, net, color_tensor):
    with torch.no_grad():
        feats1, f1_lowdim = net(img.cuda())
    sfeats1 = query_feature
    attn_intra = torch.einsum(
        "nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1)
    )
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)
    heatmap_intra = (
        F.interpolate(attn_intra, img.shape[2:], mode="bilinear", align_corners=True)
        .squeeze(0)
        .detach()
        .cpu()
    )
    img_crf = img.squeeze()
    crf_result = dense_crf(img_crf, heatmap_intra)
    heatmap_intra = torch.from_numpy(crf_result)
    heatmap = heatmap_intra.argmax(dim=0)
    return heatmap, heatmap_intra


def see_image(data, heatmap, savepath, heatmap_intra):
    data = data[0, :, :, :]
    data = data.cpu().numpy()
    data = np.clip((data * i_std + i_m) * 255, 0, 255).astype(np.uint8)
    data = data.transpose(1, 2, 0)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{savepath}/img.jpg", data)

    for i in range(heatmap_intra.shape[0]):
        heat = heatmap_intra[i, :, :].cpu().numpy()
        heat = np.round(heat * 128).astype(np.uint8)
        cv2.imwrite(f"{savepath}/heatresult{i}.jpg", heat)

class MedAD(nn.Module):

    def __init__(self, image_size=224, use_clip=True, use_dino=True, use_text=True, clip_model_name="ViT-B-32") -> None:
        super().__init__()

        clip_name = clip_model_name
        self.image_size = image_size
        pretrained = "openai"
        device = torch.device("cuda")
        self.out_layers = [3, 6, 9, 12]
        
        # 消融实验控制参数
        self.use_clip = use_clip
        self.use_dino = use_dino
        self.use_text = use_text
        
        # 根据CLIP模型设置patch size和特征维度
        if "32" in clip_name:
            self.clip_patch_size = 32
            self.clip_feature_dim = 768  # ViT-B-32
        elif "16" in clip_name:
            self.clip_patch_size = 16
            self.clip_feature_dim = 768  # ViT-B-16
        elif "14" in clip_name:
            self.clip_patch_size = 14
            if "L" in clip_name:
                self.clip_feature_dim = 1024  # ViT-L-14
            elif "H" in clip_name:
                self.clip_feature_dim = 1280  # ViT-H-14
            else:
                self.clip_feature_dim = 768  # ViT-B-14
        else:
            self.clip_patch_size = 32
            self.clip_feature_dim = 768  # 默认值

        # 根据消融实验参数选择性初始化模型
        if self.use_clip:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_name, self.image_size, pretrained=pretrained
            )  # CLIP
            self.clip_model.to(device)
            self.clip_model.eval()
            self.tokenizer = open_clip.get_tokenizer(clip_name)
        else:
            self.clip_model = None
            self.tokenizer = None

        if self.use_dino:
            self.dino_net = DinoFeaturizer()
            self.dinov2_net = torch.hub.load(
                "./models/dinov2", "dinov2_vitg14", pretrained=True, source="local"
            ).to(device)
        else:
            self.dino_net = None
            self.dinov2_net = None

        self.cfa = CFA()
        self.device = device

        self.transform_clip = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )

        self.transform_dino = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ],
        )

        self.just_resize = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size)),
            ],
        )

        transform_ce_clip = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        transform_ce_dino = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD
                ),
            ]
        )


        self.config = {}
        self.config["com_config"] = {}
        self.config["com_config"]["transform_clip"] = transform_ce_clip
        self.config["com_config"]["transform_dino"] = transform_ce_dino

        self.component_feature_extractor = ComponentFeatureExtractor(
            self.config["com_config"], clip_model=self.clip_model, dino_model=self.dinov2_net
        )

        self.decoder = LinearLayer()

        # 根据消融实验参数选择性初始化文本特征
        if self.use_text and self.use_clip:
            with torch.no_grad():
                self.text_prompts = encode_text_with_prompt_ensemble(
                    self.clip_model, ["object"], self.tokenizer, self.device
                )
        else:
            self.text_prompts = None

    def forward(
        self, batch: torch.Tensor, image_path, image_pil=None
    ) -> dict[str, torch.Tensor]:

        # 根据消融实验参数选择性进行特征提取
        image_features = None
        patch_tokens = None
        dino_patch_tokens = None
        text_features = None
        
        if self.use_clip:
            clip_transformed_image = self.transform_clip(batch)
            with torch.no_grad():
                image_features, patch_tokens = self.clip_model.encode_image(
                    clip_transformed_image, self.out_layers
                )
                image_features = image_features[:, 0, :]
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                patch_tokens = self.decoder(patch_tokens)
        
        if self.use_dino:
            dino_transformed_image = self.transform_dino(batch)
            with torch.no_grad():
                dino_patch_tokens = self.dinov2_net.forward_features(
                    dino_transformed_image
                )["x_norm_patchtokens"]
        
        if self.use_text and self.text_prompts is not None:
            # 兼容按类别动态生成的文本先验，优先使用当前类别，其次回退到 "object"
            if hasattr(self, "class_name") and isinstance(self.class_name, str) and self.class_name in self.text_prompts:
                text_features = self.text_prompts[self.class_name]
            elif "object" in self.text_prompts:
                text_features = self.text_prompts["object"]
            else:
                # 极端兜底：取第一个可用的键
                first_key = next(iter(self.text_prompts.keys()))
                text_features = self.text_prompts[first_key]

        # 先计算文本-视觉分支（zero-shot 亦可用）
        anomaly_map_vls = []
        if self.use_text and text_features is not None and patch_tokens is not None:
            for layer in range(len(patch_tokens)):
                if layer != 6:  # layer%2!=0:# (layer+1)//2!=0:
                    continue

                patch_tokens[layer] = patch_tokens[layer] @ self.clip_model.visual.proj
                patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(
                    dim=-1, keepdim=True
                )
                anomaly_map_vl = 100.0 * patch_tokens[layer] @ text_features
                B, L, C = anomaly_map_vl.shape
                H = int(np.sqrt(L))
                anomaly_map_vl = F.interpolate(
                    anomaly_map_vl.permute(0, 2, 1).view(B, 2, H, H),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map_vl = torch.softmax(anomaly_map_vl, dim=1)
                anomaly_map_vl = (
                    anomaly_map_vl[:, 1, :, :] - anomaly_map_vl[:, 0, :, :] + 1
                ) / 2
                anomaly_map_vls.append(anomaly_map_vl)
            if anomaly_map_vls:
                anomaly_map_vls = torch.mean(
                    torch.stack(anomaly_map_vls, dim=0), dim=0
                ).unsqueeze(1)
            else:
                anomaly_map_vls = torch.zeros((1, 1, self.image_size, self.image_size)).to(self.device)
        else:
            anomaly_map_vls = torch.zeros((1, 1, self.image_size, self.image_size)).to(self.device)

        # Zero-shot 路径：不依赖任何 normal 先验，仅返回文本-视觉热图
        if hasattr(self, "zero_shot") and self.zero_shot:
            if "HIS" in image_path:
                return {
                    "pred_score": torch.tensor(anomaly_map_vls.mean().item(), device=anomaly_map_vls.device),
                    "pred_mask": anomaly_map_vls,
                }
            else:
                return {
                    "pred_score": torch.tensor(anomaly_map_vls.max().item(), device=anomaly_map_vls.device),
                    "pred_mask": anomaly_map_vls,
                }

        # 计算全局相似度（需要CLIP特征）
        if self.use_clip and image_features is not None and hasattr(self, 'normal_image_features'):
            global_score = (
                1
                - (image_features @ self.normal_image_features.transpose(-2, -1))
                .max()
                .item()
            )
        else:
            global_score = 0.0

        # 计算CLIP patch相似度
        if self.use_clip and patch_tokens is not None and hasattr(self, 'normal_patch_tokens'):
            sims = []
            for i in range(len(patch_tokens)):
                if i % 2 == 0:
                    continue
                patch_tokens_reshaped = patch_tokens[i].view(
                    int((self.image_size / self.clip_patch_size) ** 2), 1, self.clip_feature_dim
                )
                normal_tokens_reshaped = self.normal_patch_tokens[i].reshape(1, -1, self.clip_feature_dim)
                cosine_similarity_matrix = F.cosine_similarity(
                    patch_tokens_reshaped, normal_tokens_reshaped, dim=2
                )
                sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                sims.append(sim_max)
            if sims:
                sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(
                    1, 1, int(self.image_size / self.clip_patch_size), int(self.image_size / self.clip_patch_size)
                )
                sim = F.interpolate(
                    sim, size=self.image_size, mode="bilinear", align_corners=True
                )
                anomaly_map_ret = 1 - sim
            else:
                anomaly_map_ret = torch.zeros((1, 1, self.image_size, self.image_size)).to(self.device)
        else:
            anomaly_map_ret = torch.zeros((1, 1, self.image_size, self.image_size)).to(self.device)

        # 计算DINO相似度
        if self.use_dino and dino_patch_tokens is not None and hasattr(self, 'normal_dino_patches'):
            dino_patch_tokens_reshaped = dino_patch_tokens.view(-1, 1, 1536)
            dino_normal_tokens_reshaped = self.normal_dino_patches.reshape(1, -1, 1536)
            cosine_similarity_matrix = F.cosine_similarity(
                dino_patch_tokens_reshaped, dino_normal_tokens_reshaped, dim=2
            )
            sim_max_dino, _ = torch.max(cosine_similarity_matrix, dim=1)
            sim_max_dino = sim_max_dino.reshape(
                1, 1, int(self.image_size / 14), int(self.image_size / 14)
            )
            sim_max_dino = F.interpolate(
                sim_max_dino, size=self.image_size, mode="bilinear", align_corners=True
            )
            anomaly_map_ret_dino = 1 - sim_max_dino
        else:
            anomaly_map_ret_dino = torch.zeros((1, 1, self.image_size, self.image_size)).to(self.device)

        # anomaly_map_vls 已在前面计算

        if self.gate == object_type.TEXTURE:

            anomaly_map_ret_all = (
                anomaly_map_ret + anomaly_map_ret_dino + anomaly_map_vls
            ) / 3

            if "HIS" in image_path:
                return {
                    "pred_score": torch.tensor(anomaly_map_ret_all.mean().item(), device=anomaly_map_ret_all.device),
                    "pred_mask": anomaly_map_ret_all,
                }
            else:
                return {
                    "pred_score": torch.tensor(anomaly_map_ret_all.max().item(), device=anomaly_map_ret_all.device),
                    "pred_mask": anomaly_map_ret_all,
                }

        # 构建查询掩膜路径，兼容多种图像格式
        query_sam_mask_path = "./masks/" + image_path.split('/data/')[-1]
        for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
            if query_sam_mask_path.endswith(ext):
                query_sam_mask_path = query_sam_mask_path.replace(ext, "/grounding_mask.png")
                break
        # print(query_sam_mask_path)
        query_tmp_mask = np.array(
            Image.open(query_sam_mask_path).resize((self.image_size, self.image_size))
        )
        query_sam_masks = split_masks_from_one_mask_torch(torch.tensor(query_tmp_mask, device=self.device))
        if len(query_sam_masks) == 0:
            query_sam_masks = [torch.ones((self.image_size, self.image_size))]

        if self.gate == object_type.SINGLE:

            anomaly_map_ret_part = torch.zeros(
                (1, 1, int(self.image_size / self.clip_patch_size), int(self.image_size / self.clip_patch_size))
            ).to(self.device)

            for query_sam_mask in query_sam_masks:
                H, W = query_sam_mask.shape
                kernel = np.ones((5, 5), np.uint8)
                # Ensure mask is on CPU numpy before OpenCV ops
                if isinstance(query_sam_mask, torch.Tensor):
                    query_sam_mask_np = (query_sam_mask.detach().cpu().numpy() > 0).astype(np.uint8)
                else:
                    query_sam_mask_np = (np.array(query_sam_mask) > 0).astype(np.uint8)
                query_sam_mask = cv2.dilate(query_sam_mask_np, kernel, iterations=1)
                thresh = torch.tensor(query_sam_mask, device=self.device, dtype=torch.float32).reshape(1, 1, H, W)
                thresh = F.interpolate(
                    thresh,
                    size=int(self.image_size / self.clip_patch_size),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(int((self.image_size / self.clip_patch_size) ** 2))
                thresh[thresh > 0] = 1
                thresh_bool = thresh > 0

                # Create separate mask for DINO features (patch size 14)
                thresh_dino = torch.tensor(query_sam_mask, device=self.device, dtype=torch.float32).reshape(1, 1, H, W)
                thresh_dino = F.interpolate(
                    thresh_dino,
                    size=int(self.image_size / 14),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(int((self.image_size / 14) ** 2))
                thresh_dino[thresh_dino > 0] = 1
                thresh_dino_bool = thresh_dino > 0

                sims = []
                for i in range(len(patch_tokens)):
                    if i % 2 == 0:
                        continue
                    patch_tokens_reshaped = patch_tokens[i].view(
                        int((self.image_size / self.clip_patch_size) ** 2), 1, self.clip_feature_dim
                    )[thresh_bool]
                    normal_tokens_reshaped = self.normal_clip_part_patch_features[i][
                        0
                    ].reshape(1, -1, self.clip_feature_dim)
                    cosine_similarity_matrix = F.cosine_similarity(
                        patch_tokens_reshaped, normal_tokens_reshaped, dim=2
                    )
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                    # print(sim_max.max())
                    sims.append(sim_max)
                sim = torch.mean(torch.stack(sims, dim=0), dim=0)

                anomaly_map_ret_dino_part = torch.zeros(
                    (1, 1, int(self.image_size / 14), int(self.image_size / 14))
                ).to(self.device)
                dino_patch_tokens_reshaped = dino_patch_tokens.view(-1, 1, 1536)[
                    thresh_dino_bool
                ]
                dino_normal_tokens_reshaped = self.normal_dino_part_patch_features[
                    0
                ].reshape(1, -1, 1536)
                cosine_similarity_matrix = F.cosine_similarity(
                    dino_patch_tokens_reshaped, dino_normal_tokens_reshaped, dim=2
                )
                sim_max_dino, _ = torch.max(cosine_similarity_matrix, dim=1)

                thresh = thresh.reshape(
                    (1, 1, int(self.image_size / self.clip_patch_size), int(self.image_size / self.clip_patch_size))
                )
                thresh_dino = thresh_dino.reshape(
                    (1, 1, int(self.image_size / 14), int(self.image_size / 14))
                )
                thresh_bool_reshaped = thresh > 0
                thresh_dino_bool_reshaped = thresh_dino > 0

                anomaly_map_ret_part[thresh_bool_reshaped] += 1 - sim
                anomaly_map_ret_dino_part[thresh_dino_bool_reshaped] += 1 - sim_max_dino

            anomaly_map_ret_part = F.interpolate(
                anomaly_map_ret_part,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map_ret_dino_part = F.interpolate(
                anomaly_map_ret_dino_part,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )

            anomaly_map_ret_all = (
                (anomaly_map_ret + anomaly_map_ret_dino) / 2
                + (anomaly_map_ret_part + anomaly_map_ret_dino_part) / 2
                + anomaly_map_vls
            ) / 3

        if self.gate == object_type.MULTI:

            heatmap, heatmap_intra = get_heatmaps(
                dino_transformed_image,
                self.train_features_sampled,
                self.dino_net,
                self.color_tensor,
            )

            # query_sam_masks = torch.tensor(query_sam_masks)
            gs_masks = torch.stack(query_sam_masks)
            cluster_masks = torch.stack(split_masks_from_one_mask_torch(heatmap + 1))

            heatmap_refined = assign_fine_to_coarse_torch(cluster_masks, gs_masks)
            heatmap_refined = heatmap_refined.max(dim=0).values

            savepath = f"./heat_masks/{self.class_name}_heat/test/0"
            if not os.path.exists(savepath):
                os.makedirs(savepath)

            cv2.imwrite(f"{savepath}/heatresult_refined.png", heatmap_refined.detach().cpu().numpy())

            anomaly_map_dist = torch.zeros((1, 1, self.image_size, self.image_size)).to(
                self.device
            )

            query_mask_path = (
                f"./heat_masks/{self.class_name}_heat/test/0/heatresult_refined.png"
            )
            query_tmp_mask = cv2.imread(query_mask_path, cv2.IMREAD_GRAYSCALE)
            query_tmp_mask = cv2.resize(query_tmp_mask, (self.image_size, self.image_size))

            # query_tmp_mask = torch.tensor(query_tmp_mask)

            query_masks_capm, query_mask_idxs = split_masks_from_one_mask_with_bg(
                query_tmp_mask
            )

            # query_masks_capm = query_masks_capm

            query_masks, _ = split_masks_from_one_mask(query_tmp_mask)

            # query_masks = query_masks

            kernel = np.ones((5, 5), np.uint8)
            query_masks_capm = [
                cv2.dilate(mask, kernel, iterations=1) for mask in query_masks_capm
            ]
            query_masks = [
                cv2.dilate(mask, kernel, iterations=1) for mask in query_masks
            ]

            anomaly_map_ret_part = torch.zeros(
                (1, 1, int(self.image_size / self.clip_patch_size), int(self.image_size / self.clip_patch_size))
            ).to(self.device)
            anomaly_map_ret_part = 100 + anomaly_map_ret_part

            anomaly_map_ret_dino_part = torch.zeros(
                (1, 1, int(self.image_size / 14), int(self.image_size / 14))
            ).to(self.device)
            anomaly_map_ret_dino_part = 100 + anomaly_map_ret_dino_part

            for j in range(len(query_masks_capm)):
                query_sam_mask = query_masks_capm[j]
                H, W = query_sam_mask.shape
                thresh = torch.tensor(query_sam_mask, device=self.device, dtype=torch.float32).reshape(1, 1, H, W)
                thresh = F.interpolate(
                    thresh,
                    size=int(self.image_size / self.clip_patch_size),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(int((self.image_size / self.clip_patch_size) ** 2))

                if thresh.sum() < 1:
                    continue

                thresh[thresh > 0] = 1
                thresh_bool = thresh > 0

                # Create separate mask for DINO features (patch size 14)
                thresh_dino = torch.tensor(query_sam_mask, device=self.device, dtype=torch.float32).reshape(1, 1, H, W)
                thresh_dino = F.interpolate(
                    thresh_dino,
                    size=int(self.image_size / 14),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(int((self.image_size / 14) ** 2))
                thresh_dino[thresh_dino > 0] = 1
                thresh_dino_bool = thresh_dino > 0

                if self.normal_dino_part_patch_features[query_mask_idxs[j]] == []:
                    continue

                sims = []
                for i in range(len(patch_tokens)):
                    if i % 2 == 0:  # (layer+1)//2!=0:
                        continue
                    patch_tokens_reshaped = patch_tokens[i].view(
                        int((self.image_size / self.clip_patch_size) ** 2), 1, self.clip_feature_dim
                    )[thresh_bool]
                    normal_tokens_reshaped = self.normal_clip_part_patch_features[i][
                        query_mask_idxs[j]
                    ].reshape(1, -1, self.clip_feature_dim)
                    cosine_similarity_matrix = F.cosine_similarity(
                        patch_tokens_reshaped, normal_tokens_reshaped, dim=2
                    )
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                    sims.append(sim_max)
                sim = torch.mean(torch.stack(sims, dim=0), dim=0)

                dino_patch_tokens_reshaped = dino_patch_tokens.view(-1, 1, 1536)[
                    thresh_dino_bool
                ]
                dino_normal_tokens_reshaped = self.normal_dino_part_patch_features[
                    query_mask_idxs[j]
                ].reshape(1, -1, 1536)
                cosine_similarity_matrix = F.cosine_similarity(
                    dino_patch_tokens_reshaped, dino_normal_tokens_reshaped, dim=2
                )
                sim_max_dino, _ = torch.max(cosine_similarity_matrix, dim=1)
                thresh = thresh.reshape(
                    (1, 1, int(self.image_size / self.clip_patch_size), int(self.image_size / self.clip_patch_size))
                )
                thresh_dino = thresh_dino.reshape(
                    (1, 1, int(self.image_size / 14), int(self.image_size / 14))
                )
                thresh_bool_reshaped = thresh > 0
                thresh_dino_bool_reshaped = thresh_dino > 0
                
                anomaly_map_ret_part[thresh_bool_reshaped] = torch.min(
                    1 - sim, anomaly_map_ret_part[thresh_bool_reshaped]
                )
                anomaly_map_ret_dino_part[thresh_dino_bool_reshaped] = torch.min(
                    1 - sim_max_dino, anomaly_map_ret_dino_part[thresh_dino_bool_reshaped]
                )

            anomaly_map_ret_part[anomaly_map_ret_part == 100] = 0
            anomaly_map_ret_dino_part[anomaly_map_ret_dino_part == 100] = 0

            anomaly_map_ret_part = F.interpolate(
                anomaly_map_ret_part,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map_ret_dino_part = F.interpolate(
                anomaly_map_ret_dino_part,
                size=self.image_size,
                mode="bilinear",
                align_corners=True,
            )

            if image_pil is not None:
                image = np.array(image_pil[0])
                # Resize image to match self.image_size if needed
                if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                    image = cv2.resize(image, (self.image_size, self.image_size))
            else:
                image = np.array(
                    Image.open(image_path)
                    .convert("RGB")
                    .resize((self.image_size, self.image_size))
                )

            features = self.component_feature_extractor.extract(image, query_masks)

            query_component_feats = {
                "area": [],
                "color": [],
                "position": [],
                "clip_image": [],
                "dino_image": [],
                "geo": [],
            }

            for feature_name in ["area", "color", "position", "clip_image", "dino_image"]:
                query_component_feats[feature_name].append(features[feature_name])

            for feature_name in ["area", "color", "position", "clip_image", "dino_image"]:
                query_component_feats[feature_name] = torch.cat(
                    query_component_feats[feature_name], axis=0
                )

            query_component_feats["geo"] = torch.cat(
                [
                    query_component_feats["area"],
                    query_component_feats["color"],
                    query_component_feats["position"],
                ],
                dim=1,
            )
            query_component_feats["clip_image"] = query_component_feats[
                "clip_image"
            ].transpose(0, 1)

            # query_component_feats["dino_image"] = query_component_feats[
            #     "dino_image"
            # ].transpose(0, 1)

            for layer in range(query_component_feats["clip_image"].shape[0]):
                query_component_feats["clip_image"][layer] = self.cfa(
                    query_component_feats["clip_image"][layer]
                )

            query_component_feats["dino_image"] = self.cfa(
                query_component_feats["dino_image"]
            )

            for mask_idx in range(len(query_masks)):
                query_mask = query_masks[mask_idx]
                thresh_ori = torch.tensor(query_mask, device=self.device).reshape(
                    1, 1, self.image_size, self.image_size
                )
                sim_clip_component = F.cosine_similarity(
                    query_component_feats["clip_image"][:, mask_idx, :]
                    .unsqueeze(1)
                    .unsqueeze(1),
                    self.normal_component_feats["clip_image"].unsqueeze(1),
                    dim=-1,
                )

                # print( query_component_feats["dino_image"][mask_idx].shape, self.normal_component_feats["dino_image"].shape)

                sim_dino_component = F.cosine_similarity(
                    query_component_feats["dino_image"][mask_idx].unsqueeze(0),
                    self.normal_component_feats["dino_image"],
                    dim=1,
                )

                # print(sim_dino_component.shape)

                sim_geo = F.cosine_similarity(
                    query_component_feats["geo"][mask_idx],
                    self.normal_component_feats["geo"].unsqueeze(0),
                    dim=2,
                )
                dist = torch.mean(
                    1 - sim_clip_component.max(dim=-1).values, dim=0
                ).item()
                dist += 1 - sim_dino_component.max().item()
                dist += 1 - sim_geo.max().item()
                thresh_ori_bool = thresh_ori > 0
                anomaly_map_dist[thresh_ori_bool] += dist

            anomaly_map_ret_all = (
                (anomaly_map_ret + anomaly_map_ret_dino) / 2
                + (anomaly_map_ret_part + anomaly_map_ret_dino_part) / 2
                + anomaly_map_vls
            ) / 3 + anomaly_map_dist / 2

        return {
            "pred_score": torch.tensor(anomaly_map_ret_all.max().item() + global_score, device=anomaly_map_ret_all.device),
            "pred_mask": anomaly_map_ret_all,
        }

    def setup(self, data: dict, re_seg=True) -> None:

        few_shot_samples = data.get("few_shot_samples")
        self.class_name = data.get("dataset_category")
        image_paths = data.get("image_path")

        # 根据类别动态设置文本先验（医疗类将自动使用医疗模板）
        if self.use_text and self.use_clip:
            try:
                self.text_prompts = encode_text_with_prompt_ensemble(
                    self.clip_model, [self.class_name], self.tokenizer, self.device
                )
            except Exception:
                # 回退到通用对象词
                self.text_prompts = encode_text_with_prompt_ensemble(
                    self.clip_model, ["object"], self.tokenizer, self.device
                )
        else:
            self.text_prompts = None

        # 打印当前类别使用的文本 prompts（显示总数与前若干条示例，避免过长输出）
        try:
            prompt_examples = get_prompt_sentences_for_object(self.class_name)
        except Exception:
            prompt_examples = get_prompt_sentences_for_object("object")
        max_show = 8
        print(f"[TextPrompts][{self.class_name}] total={len(prompt_examples)}; examples={prompt_examples[:max_show]}")

        self.kernel = np.ones((20, 20), np.uint8)
        # 兼容 zero-shot：few_shot_samples 可能为空/None
        zero_shot = False
        if few_shot_samples is None:
            self.shot = 0
            zero_shot = True
        elif isinstance(few_shot_samples, torch.Tensor) and few_shot_samples.numel() == 0:
            self.shot = 0
            zero_shot = True
        elif isinstance(few_shot_samples, (list, tuple)) and len(few_shot_samples) == 0:
            self.shot = 0
            zero_shot = True
        else:
            # 保证为 tensor
            if isinstance(few_shot_samples, (list, tuple)):
                few_shot_samples = torch.cat(few_shot_samples, dim=0)
            self.shot = len(few_shot_samples)

        if zero_shot:
            self.zero_shot = True
            # zero-shot 下无需构建任何 normal 先验，设置一个默认 gate
            self.gate = object_type.TEXTURE
            print(f"[MedAD.setup] class={self.class_name}, gate={self.gate.name}, zero_shot=True")
            return

        clip_transformed_normal_image = self.transform_clip(few_shot_samples).to(
            self.device
        )
        dino_transformed_normal_image = self.transform_dino(few_shot_samples).to(
            self.device
        )

        self.part_num = {
            "breakfast_box": [4],
            "screw_bag": [3],
            "splicing_connectors": [2],
            "pushpins": [3],
            "juice_bottle": [4],
        }

        num_cluster = {
            "breakfast_box": 5,
            "screw_bag": 5,
            "splicing_connectors": 5,
            "pushpins": 5,
            "juice_bottle": 5,
        }

        color_list = [
            [0, 0, 0],
            [127, 123, 229],
            [195, 240, 251],
            [146, 223, 255],
            [243, 241, 230],
            [224, 190, 144],
            [178, 116, 75],
        ]
        color_tensor = torch.tensor(color_list, device=self.device)
        color_tensor = color_tensor[:, :, None, None]
        self.color_tensor = color_tensor.repeat(1, 1, self.image_size, self.image_size)

        grounded_sam_mask_paths = []
        for image_path in image_paths:
            # 构建掩膜路径，兼容多种图像格式
            mask_path = "./masks/" + image_path.split('/data/')[-1]
            # 替换所有可能的图像扩展名为掩膜路径
            for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
                if mask_path.endswith(ext):
                    mask_path = mask_path.replace(ext, "/grounding_mask.png")
                    break
            grounded_sam_mask_paths.append(mask_path)
        grounded_sam_masks = [
            split_masks_from_one_mask_torch(
                torch.tensor(
                    np.array(Image.open(x).resize((self.image_size, self.image_size)))
                )
            )
            for x in grounded_sam_mask_paths
        ]

        if len(grounded_sam_masks[0]) > 0:
            H, W = grounded_sam_masks[0][0].shape
            object_ratio = (torch.sum(sorted(grounded_sam_masks[0], key=lambda x:torch.sum(x), reverse=True)[0]) / 255) / (H * W)
        else:
            object_ratio = 1

        if object_ratio > 0.65 and len(grounded_sam_masks[0]) <= 2:
            self.gate = object_type.TEXTURE
        elif len(grounded_sam_masks[0]) == 1:
            self.gate = object_type.SINGLE
        else:
            self.gate = object_type.MULTI

        # 打印当前类别与分流gate类型，便于运行时查看
        print(f"[MedAD.setup] class={self.class_name}, gate={self.gate.name}")

        with torch.no_grad():
            # 根据消融实验参数选择性提取正常样本特征
            if self.use_clip:
                self.normal_image_features, self.normal_patch_tokens = (
                    self.clip_model.encode_image(
                        clip_transformed_normal_image, self.out_layers
                    )
                )
                self.normal_image_features = self.normal_image_features[:, 0, :]
                self.normal_image_features = (
                    self.normal_image_features / self.normal_image_features.norm()
                )
                self.normal_patch_tokens = self.decoder(self.normal_patch_tokens)
            else:
                self.normal_image_features = None
                self.normal_patch_tokens = None

            if self.use_dino:
                self.normal_dino_patches = self.dinov2_net.forward_features(
                    dino_transformed_normal_image
                )["x_norm_patchtokens"]
            else:
                self.normal_dino_patches = None

        self.normal_dino_part_patch_features = [[], [], [], [], [], [], [], [], [], []]

        self.normal_clip_part_patch_features = [
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], [], [], []],
        ]

        if self.gate == object_type.SINGLE:

            for i in range(self.shot):
                # Ensure mask is on CPU numpy before OpenCV ops
                mask = grounded_sam_masks[i][0]
                if isinstance(mask, torch.Tensor):
                    mask_np = (mask.detach().cpu().numpy() > 0).astype(np.uint8)
                else:
                    mask_np = (np.array(mask) > 0).astype(np.uint8)
                normal_sam_mask = cv2.dilate(
                    mask_np, self.kernel, iterations=1
                )
                thresh = torch.tensor(normal_sam_mask, device=self.device, dtype=torch.float32).reshape(1, 1, H, W)
                thresh = F.interpolate(
                    thresh,
                    size=int(self.image_size / self.clip_patch_size),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(int((self.image_size / self.clip_patch_size) ** 2))
                thresh[thresh > 0] = 1
                thresh_bool = thresh > 0

                # Create separate mask for DINO features (patch size 14)
                thresh_dino = torch.tensor(normal_sam_mask, device=self.device, dtype=torch.float32).reshape(1, 1, H, W)
                thresh_dino = F.interpolate(
                    thresh_dino,
                    size=int(self.image_size / 14),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(int((self.image_size / 14) ** 2))
                thresh_dino[thresh_dino > 0] = 1
                thresh_dino_bool = thresh_dino > 0

                selected_patch_features = self.normal_dino_patches[i][thresh_dino_bool]
                self.normal_dino_part_patch_features[0].append(selected_patch_features)

                for layer in range(len(self.normal_patch_tokens)):
                    if layer % 2 == 0:
                        continue
                    selected_patch_features = self.normal_patch_tokens[layer][i][thresh_bool]
                    self.normal_clip_part_patch_features[layer][0].append(
                        selected_patch_features
                    )

            self.normal_dino_part_patch_features[0] = torch.cat(
                self.normal_dino_part_patch_features[0], dim=0
            )
            for layer in range(len(self.normal_patch_tokens)):
                if layer % 2 == 0:
                    continue
                self.normal_clip_part_patch_features[layer][0] = torch.cat(
                    self.normal_clip_part_patch_features[layer][0], dim=0
                )

        if self.gate == object_type.MULTI:

            if re_seg:
                train_feature_list = []
                greedsampler_perimg = GreedyCoresetSampler(
                    percentage=0.01, device="cuda"
                )

                for Img in dino_transformed_normal_image:
                    Img = Img.unsqueeze(0)
                    feats0, f_lowdim = self.dino_net(Img)
                    feats = feats0.squeeze()
                    feats = feats.reshape(feats0.shape[1], -1).permute(1, 0)
                    feats_sample = greedsampler_perimg.run(feats)
                    train_feature_list.append(feats_sample)

                train_features = torch.cat(train_feature_list, dim=0)
                train_features = F.normalize(train_features, dim=1)
                train_features = train_features.cpu().numpy()

                part_num = -1
                if self.class_name in self.part_num.keys():
                    part_num_right = self.part_num[self.class_name]
                    n_cluster = num_cluster[self.class_name]
                else:
                    part_num_right = [1]
                    n_cluster = 2

                while part_num not in part_num_right:
                    kmeans = KMeans(init="k-means++", n_clusters=n_cluster)
                    c = kmeans.fit(train_features)
                    cluster_centers = torch.from_numpy(c.cluster_centers_)
                    train_features_sampled = cluster_centers.to(self.device)
                    train_features_sampled = train_features_sampled.unsqueeze(
                        0
                    ).unsqueeze(0)
                    self.train_features_sampled = train_features_sampled.permute(
                        0, 3, 1, 2
                    )

                    for i, Img in enumerate(dino_transformed_normal_image):
                        Img = Img.unsqueeze(0)
                        heatmap, heatmap_intra = get_heatmaps(
                            Img,
                            self.train_features_sampled,
                            self.dino_net,
                            self.color_tensor,
                        )

                        savepath = f"./heat_masks/{self.class_name}_heat/train/{i}"
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)

                        gs_masks = torch.stack(grounded_sam_masks[i])
                        cluster_masks = torch.stack(
                            split_masks_from_one_mask_torch(heatmap + 1)
                        )
                        heatmap_refined = assign_fine_to_coarse_torch(
                            cluster_masks, gs_masks
                        )
                        heatmap_refined = heatmap_refined.max(dim=0).values

                        cv2.imwrite(
                            f"{savepath}/heatresult_refined.png",
                            heatmap_refined.detach().cpu().numpy(),
                        )
                        see_image(Img, heatmap, f"{savepath}", heatmap_intra)
                        part_num = len(filter_bg_noise("./heat_masks", self.class_name))

                        plt.clf()
                        plt.imshow(heatmap_refined.detach().cpu().numpy())
                        plt.savefig(savepath + "/" + "masks_color.png")

                    torch.save(
                        self.train_features_sampled,
                        f"./heat_masks/{self.class_name}_heat/train_features_sampled.pth",
                    )
            else:
                self.train_features_sampled = torch.load(
                    f"./heat_masks/{self.class_name}_heat/train_features_sampled.pth"
                )

            self.normal_component_feats = {
                "area": [],
                "color": [],
                "position": [],
                "clip_image": [],
                "dino_image": [],
                "geo": [],
            }

            for i in range(self.shot):
                image = np.array(
                    Image.open(image_paths[i])
                    .convert("RGB")
                    .resize((self.image_size, self.image_size))
                )
                normal_mask_path = f"./heat_masks/{self.class_name}_heat/train/{i}/heatresult_refined.png"
                normal_masks = cv2.imread(normal_mask_path, cv2.IMREAD_GRAYSCALE)

                normal_masks_capm, normal_mask_idxs = split_masks_from_one_mask_with_bg(
                    normal_masks
                )

                normal_masks, _ = split_masks_from_one_mask(normal_masks)

                normal_masks_capm = [
                    cv2.dilate(mask, self.kernel, iterations=1)
                    for mask in normal_masks_capm
                ]

                kernel = np.ones((5, 5), np.uint8)
                normal_masks = [
                    cv2.dilate(mask, kernel, iterations=1) for mask in normal_masks
                ]

                for j in range(len(normal_mask_idxs)):
                    thresh = torch.tensor(normal_masks_capm[j], device=self.device, dtype=torch.float32).reshape(1, 1, H, W)
                    thresh = F.interpolate(
                        thresh,
                        size=int(self.image_size / self.clip_patch_size),
                        mode="bilinear",
                        align_corners=True,
                    ).reshape(int((self.image_size / self.clip_patch_size) ** 2))

                    if thresh.sum() < 1:
                        continue
                    thresh[thresh > 0] = 1
                    thresh_bool = thresh > 0

                    # Create separate mask for DINO features (patch size 14)
                    thresh_dino = torch.tensor(normal_masks_capm[j], device=self.device, dtype=torch.float32).reshape(1, 1, H, W)
                    thresh_dino = F.interpolate(
                        thresh_dino,
                        size=int(self.image_size / 14),
                        mode="bilinear",
                        align_corners=True,
                    ).reshape(int((self.image_size / 14) ** 2))
                    thresh_dino[thresh_dino > 0] = 1
                    thresh_dino_bool = thresh_dino > 0

                    selected_patch_features = self.normal_dino_patches[i][thresh_dino_bool]
                    self.normal_dino_part_patch_features[normal_mask_idxs[j]].append(
                        selected_patch_features
                    )

                    for layer in range(len(self.normal_patch_tokens)):
                        if layer % 2 == 0:
                            continue
                        selected_patch_features = self.normal_patch_tokens[layer][i][
                            thresh_bool
                        ]
                        self.normal_clip_part_patch_features[layer][
                            normal_mask_idxs[j]
                        ].append(selected_patch_features)

                features = self.component_feature_extractor.extract(
                    image,
                    normal_masks,
                )
                for feature_name in ["area", "color", "position", "clip_image", "dino_image"]:
                    self.normal_component_feats[feature_name].append(
                        features[feature_name]
                    )

            for j in range(len(normal_mask_idxs)):
                if self.normal_dino_part_patch_features[normal_mask_idxs[j]] == []:
                    continue
                self.normal_dino_part_patch_features[normal_mask_idxs[j]] = torch.cat(
                    self.normal_dino_part_patch_features[normal_mask_idxs[j]], dim=0
                )

                for layer in range(len(self.normal_patch_tokens)):
                    if layer % 2 == 0:
                        continue
                    self.normal_clip_part_patch_features[layer][normal_mask_idxs[j]] = (
                        torch.cat(
                            self.normal_clip_part_patch_features[layer][
                                normal_mask_idxs[j]
                            ],
                            dim=0,
                        )
                    )

            for feature_name in ["area", "color", "position", "clip_image", "dino_image"]:
                self.normal_component_feats[feature_name] = torch.cat(
                    self.normal_component_feats[feature_name], axis=0
                )
            self.normal_component_feats["clip_image"] = (
                self.normal_component_feats["clip_image"].transpose(0, 1)
            )

            for layer in range(self.normal_component_feats["clip_image"].shape[0]):
                self.normal_component_feats["clip_image"][layer] = self.cfa(
                    self.normal_component_feats["clip_image"][layer]
                )

            self.normal_component_feats["dino_image"] = self.cfa(
                self.normal_component_feats["dino_image"]
            )


            self.normal_component_feats["geo"] = torch.cat(
                [
                    self.normal_component_feats["area"],
                    self.normal_component_feats["color"],
                    self.normal_component_feats["position"],
                ],
                dim=1,
            )


def calculate_iou_torch(mask1, mask2, threshold=0.5):
    # 转换为布尔 mask
    mask1_bin = (mask1 > threshold)
    mask2_bin = (mask2 > threshold)

    # 计算交集和并集
    intersection = torch.sum((mask1_bin & mask2_bin).float())
    union = torch.sum((mask1_bin | mask2_bin).float())

    if union == 0:
        return torch.tensor(0.0, device=mask1.device)

    return intersection / union


def assign_fine_to_coarse_torch(coarse_masks, fine_masks):
    M, H, W = coarse_masks.shape
    N = fine_masks.shape[0]

    coarse_to_fine_masks = {i: [] for i in range(M)}
    for fine_idx in range(N):
        if N > 1:
            if fine_masks[fine_idx][0, 0] and fine_masks[fine_idx][H - 1, W - 1]:
                continue
            if fine_masks[fine_idx][10, 10] and fine_masks[fine_idx][H - 10, W - 10]:
                continue
        best_iou = 0
        best_coarse_idx = -1
        for coarse_idx in range(M):
            iou = calculate_iou_torch(fine_masks[fine_idx], coarse_masks[coarse_idx])
            if iou > best_iou:
                best_iou = iou
                best_coarse_idx = coarse_idx
        if best_coarse_idx != -1:
            coarse_to_fine_masks[best_coarse_idx].append(fine_masks[fine_idx])

    new_coarse_masks = torch.zeros_like(coarse_masks)
    for coarse_idx in coarse_to_fine_masks.keys():
        assigned_fine_masks = coarse_to_fine_masks[coarse_idx]
        if len(assigned_fine_masks) > 0:
            for fine_mask in assigned_fine_masks:
                new_coarse_masks[coarse_idx][fine_mask > 0] = coarse_idx + 1

    return new_coarse_masks


class CFA(nn.Module):
    def __init__(self):
        super(CFA, self).__init__()

    def _compute_similarity_matrix(self, tensors):
        similarity_matrix = F.cosine_similarity(
            tensors.unsqueeze(1), tensors.unsqueeze(0), dim=-1
        )
        return similarity_matrix

    def _normalize_adjacency(self, adj_matrix):
        row_sum = adj_matrix.sum(dim=1, keepdim=True)
        normalized_adj = adj_matrix / row_sum
        return normalized_adj

    def forward(self, tensor):
        # print(tensor.shape)
        similarity_matrix = self._compute_similarity_matrix(tensor)
        normalized_adj_matrix = self._normalize_adjacency(similarity_matrix)
        aggregated_tensors = normalized_adj_matrix @ tensor
        return aggregated_tensors
