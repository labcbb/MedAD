import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
# segment anything
from .segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator
)
# Grounding DINO
from .grounded_sam import (
    load_image,
    load_model,
    get_grounding_output,
    show_box,
    show_mask
)

import tqdm
import os
import glob
import random
from .component_feature_extractor import ComponentFeatureExtractor
import sklearn.cluster

import scipy.stats
import torch
from PIL import Image
import yaml
import time

def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def turn_binary_to_int(mask):
    temp = np.where(mask,255,0).astype(np.uint8)
    return temp

def crop_by_mask(image,mask):
        if image.shape[-1] == 3:
            mask = cv2.merge([mask,mask,mask])
        return np.where(mask!=0,image,0)

def remove_background(background,masks):
    result = list()
    for mask in masks:
        # if np.sum(np.logical_and(mask,background)) == 0:
        # cv2.imshow(f"mask{intersect_ratio(mask,background)}",mask)
        # cv2.waitKey(0)
        if intersect_ratio(mask,background) < 0.95:
            result.append(mask)
    return result

def split_masks_by_connected_component(masks):
    result_masks = list()
    for i,mask in enumerate(masks):
        #mask_uint = turn_binary_to_int(mask['segmentation'])
        if type(mask) == dict:
            mask_uint = turn_binary_to_int(mask['segmentation'])
        else:
            mask_uint = mask
        # cv2.imshow("mask",mask_uint)
        # cv2.waitKey(0)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint,connectivity=4,ltype=None)
        for j in range(num_labels):
            area = stats[j,4]
            w,h = stats[j,2], stats[j,3]
            # if w*h < mask_uint.shape[0]*mask_uint.shape[1]*0.95 and area>50:# and area>0.01*mask_uint.shape[0]*mask_uint.shape[1]:
            # if area > 10:
                #pass
                #result_masks.append({'area':area,"segmentation":labels==j})
            result_masks.append(turn_binary_to_int(labels==j))
    return result_masks

def color_masks(masks):
    # if type(masks) != list:
    #     masks = [masks]
    if type(masks) == list and len(masks) == 1:
        return np.where(masks[0],255,0).astype(np.uint8)
    if type(masks) != list and len(masks.shape) == 2:
        return np.where(masks!=0,255,0).astype(np.uint8)
    color_mask = np.zeros([masks[0].shape[0],masks[0].shape[1],3],dtype=np.uint8)
    masks = sorted(masks,key=lambda x:np.sum(x),reverse=True)
    for i,mask in enumerate(masks):
        color_mask[mask!=0] = np.random.randint(0,255,[3])
    return color_mask

def merge_masks(masks):
    # remove empty masks
    # masks = filter_small_masks(masks,threshold=0.001)

    result_mask = np.zeros_like(masks[0],dtype=np.uint8)
    # masks = sorted(masks,key=lambda x:np.sum(x),reverse=True)
    for i,mask in enumerate(masks):
        #mask = binary_fill_holes(mask).astype(np.uint8)
        result_mask[mask!=0] = np.ones_like(mask)[mask!=0]*(i+1)
    return result_mask

def split_masks_from_one_mask(masks):
    result_masks = list()
    result_idxs = list()
    for i in range(1,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        if np.sum(mask!=0)/mask.size > 0.0001:
            result_masks.append(mask)
            result_idxs.append(i)
    return result_masks, result_idxs

def split_masks_from_one_mask_with_bg(masks):
    result_masks = list()
    result_idxs = list()
    for i in range(0,np.max(masks)+1):
        mask = np.zeros_like(masks)
        mask[masks==i] = 255
        if np.sum(mask!=0)/mask.size > 0.0001:
            result_masks.append(mask)
            result_idxs.append(i)
    return result_masks, result_idxs

def split_masks_from_one_mask_with_bg_torch(masks):
    result_masks = list()
    result_idxs = list()
    H, W = masks.shape
    for i in range(1,torch.max(masks)+1):
        mask = torch.zeros_like(masks)
        mask[masks==i] = 255
        if torch.sum(mask!=0) / (H * W) > 0.001:
            result_masks.append(mask)
            result_idxs.append(i)
    return result_masks, result_idxs

def split_masks_from_one_mask_torch(masks):
    result_masks = list()
    H, W = masks.shape
    for i in range(1,torch.max(masks)+1):
        mask = torch.zeros_like(masks)
        mask[masks==i] = 255
        if torch.sum(mask!=0) / (H * W) > 0.001:
            result_masks.append(mask)
    return result_masks

def resize_mask(mask,image_size):
    mask = cv2.resize(mask,[image_size,image_size],interpolation=cv2.INTER_LINEAR)
    mask[mask>128]=255
    mask[mask<=128]=0
    return mask

def intersect_ratio(mask1,mask2):
    intersection = np.logical_and(mask1,mask2)
    if intersection.sum() == 0:
        return 0
    ratio = np.sum(intersection)/min([np.sum(mask1!=0),np.sum(mask2!=0)])
    ratio = 0 if np.isnan(ratio) else ratio
    return ratio

def iou(mask1,mask2):
    intersection = np.logical_and(mask1,mask2)
    union = np.logical_or(mask1,mask2)
    return np.sum(intersection)/np.sum(union)

def remove_duplicate_masks(masks, iou_threshold=0.9):
    # List to store indices of masks to keep
    keep_masks = []

    for i in range(len(masks)):
        is_duplicate = False
        for j in range(len(keep_masks)):
            combine = np.hstack([masks[i],masks[keep_masks[j]]])
            combine = cv2.resize(combine,[512,256])
            cv2.imshow(f"{iou(masks[i], masks[keep_masks[j]])}",combine)
            cv2.waitKey(0)
            if iou(masks[i], masks[keep_masks[j]]) > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep_masks.append(i)

    # Filter the original list of masks
    unique_masks = [masks[i] for i in keep_masks]
    return unique_masks

def filter_small_masks(masks,threshold=0.001):
    new_masks = list()
    for mask in masks:
        if np.sum(mask!=0)/mask.size > threshold:
            new_masks.append(mask)
    return new_masks



def filter_masks_by_grounding_mask(grounding_mask,masks):
    new_mask = list()
    for mask in masks:
        if intersect_ratio(grounding_mask,mask) > 0.9:
            new_mask.append(mask)
    return new_mask

def get_padding_functions(orig_size,target_size=256,mode='linear'):
    """
        padding_func, inverse_padding_func = get_padding_functions(image.size,target_size=256)
        image2 = padding_func(image) # image2.size = (256,256) with padding
        image2.show()
        image3 = inverse_padding_func(image2) # image3.size = (256,256) without padding
        image3.show()
    """
    imsize = orig_size
    long_size = max(imsize)
    scale = target_size / long_size
    new_h = int(imsize[1] * scale + 0.5)
    new_w = int(imsize[0] * scale + 0.5)
    pad_w = int((target_size - new_w)//2)
    pad_h = int((target_size - new_h)//2)
    inter =  transforms.InterpolationMode.NEAREST if mode == 'nearest' else transforms.InterpolationMode.BILINEAR

    padding_func = transforms.Compose([
        transforms.Resize((new_h,new_w),interpolation=inter),
        transforms.Pad((pad_w,pad_h,pad_w,pad_h),fill=0,padding_mode='constant')
    ])
    inverse_padding_func = transforms.Compose([
        transforms.CenterCrop((new_h,new_w)),
        transforms.Resize((target_size,target_size),interpolation=inter)
    ])
    return padding_func, inverse_padding_func

def filter_by_combine(masks):
    #masks = split_masks_from_one_mask(merge_masks(masks))
    
    masks = filter_small_masks(masks,threshold=0.001)
    masks = sorted(masks,key=lambda x:np.sum(x)) # small to large
    # if len(masks) < 1:
    #     return masks
    combine_masks = np.zeros_like(masks[0])
    result_masks = list()
    wait_masks = list()
    for i,mask in enumerate(masks):
        if intersect_ratio(combine_masks,mask) < 0.9 or i == 0:
            combine_masks = np.logical_or(combine_masks,mask)
            result_masks.append(mask)
        else:
            wait_masks.append(mask)
    
    if len(wait_masks) != 0:
        for mask in wait_masks:
            ratio = np.sum(np.logical_and(combine_masks,mask))/np.sum(mask!=0)
            if ratio < 0.9:
                combine_masks = np.logical_or(combine_masks,mask)
                result_masks.append(mask)
    return result_masks


def grounding_segmentation(img_paths, save_path, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    box_threshold = config['box_threshold']
    text_threshold = config['text_threshold']
    text_prompt = config['text_prompt']

    # 模型只加载一次
    model = load_model(
        './models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        './pretrained_ckpts/groundingdino_swint_ogc.pth',
        device
    )
    # SAM 后端可切换（默认保持原始 SAM-HQ，不影响现有功能）
    sam_backend = config.get('sam_backend', 'sam_hq')  # 'sam_hq' | 'medsam'
    if sam_backend == 'medsam':
        # 使用 MedSAM（与 SAM Predictor 接口兼容），按配置指定架构与权重
        medsam_arch = config.get('medsam_arch', 'vit_b')
        medsam_ckpt = config.get('medsam_ckpt', './pretrained_ckpts/medsam_vit_b.pth')
        sam = sam_model_registry[medsam_arch](checkpoint=medsam_ckpt).to(device)
        predictor = SamPredictor(sam)
        print(f"[grounding_segmentation] 使用分割后端: MedSAM | arch={medsam_arch} | ckpt={medsam_ckpt}")
    else:
        # 默认：SAM-HQ ViT-H（与现有逻辑一致）
        sam = sam_hq_model_registry['vit_h']('./pretrained_ckpts/sam_hq_vit_h.pth').to(device)
        predictor = SamPredictor(sam)
        print(f"[grounding_segmentation] 使用分割后端: SAM-HQ | arch=vit_h | ckpt=./pretrained_ckpts/sam_hq_vit_h.pth")

    for image_path in tqdm.tqdm(img_paths, desc="grounding..."):
        # 避免重复读取图片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        # GroundingDINO 推理
        image_pil, image_dino = load_image(image_path)
        boxes_filt, pred_phrases = get_grounding_output(
            model, image_dino, text_prompt, box_threshold, text_threshold, device=device
        )

        # 直接批量筛选背景索引
        background_prompt = [j for j in config['background_prompt'].split('.') if j.strip()]
        background_box = [i for i, text in enumerate(pred_phrases)
                          if any(bp in text.replace(' - ', '-') for bp in background_prompt)]

        predictor.set_image(image)

        if len(boxes_filt) == 0:
            masks = np.ones((H, W), dtype=np.uint8)
            background = np.zeros_like(masks)
            color_mask = masks.copy()
        else:
            # 批量处理 bbox，不用 for 循环
            boxes_filt = boxes_filt.to(device) * torch.tensor([W, H, W, H], device=device)
            boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
            boxes_filt[:, 2:] += boxes_filt[:, :2]

            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes_filt, (H, W)
            ).to(device)

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )  # (N,1,H,W)

            # 背景 mask 在 GPU 上合并
            if len(background_box) > 0:
                background = masks[background_box].sum(dim=0).squeeze(0)  # (H,W)
                background = (background > 0).to(torch.uint8).cpu().numpy() * 255
            else:
                background = np.zeros((H, W), dtype=np.uint8)

            # 过滤非背景 mask
            keep_idx = [i for i in range(len(masks)) if i not in background_box]
            if keep_idx:
                masks = masks[keep_idx, 0].cpu().numpy()
                masks = turn_binary_to_int(masks)
                if config.get('filter_by_combine', False):
                    masks = filter_by_combine(masks)
                color_mask = color_masks(masks)
                masks = merge_masks(masks)
            else:
                masks = np.zeros((H, W), dtype=np.uint8)
                color_mask = masks.copy()

        # 保存结果 - 兼容多种图像格式
        # 移除文件扩展名，保持目录结构
        image_path_no_ext = image_path
        for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
            if image_path_no_ext.endswith(ext):
                image_path_no_ext = image_path_no_ext[:-len(ext)]
                break
        image_name = '/'.join(image_path_no_ext.split("/")[-3:])
        save_dir = f"{save_path}/{image_name}"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/grounding_mask.png", masks)
        cv2.imwrite(f"{save_dir}/grounding_background.png", background)
        cv2.imwrite(f"{save_dir}/grounding_mask_color.png", color_mask)




def segmentation(img_paths,save_path,use_grounding_filter=False,no_sam=True):
    for p in tqdm.tqdm(img_paths,desc="segmentation..."):
        # 兼容多种图像格式 - 移除文件扩展名
        p_no_ext = p
        for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
            if p_no_ext.endswith(ext):
                p_no_ext = p_no_ext[:-len(ext)]
                break
        image_name = '/'.join(p_no_ext.split("/")[-3:])
        print(image_name)
        refined_masks = cv2.imread(f"{save_path}/{image_name}/grounding_mask.png",cv2.IMREAD_GRAYSCALE)
        refined_masks = split_masks_from_one_mask(refined_masks)
        if len(refined_masks) > 0:
            refined_masks = split_masks_by_connected_component(refined_masks)
        if len(refined_masks) > 0:
            refined_masks = filter_by_combine(refined_masks)
        refined_masks_color = color_masks(refined_masks)
        if len(refined_masks) > 0:
            refined_masks = merge_masks(refined_masks)
        # cv2.imwrite(f"{save_path}/{image_name}/all_masks.png",merge_masks(raw_splited_masks))
        # cv2.imwrite(f"{save_path}/{image_name}/all_masks_color.png",color_masks(raw_splited_masks))
        cv2.imwrite(f"{save_path}/{image_name}/refined_masks.png",refined_masks)
        cv2.imwrite(f"{save_path}/{image_name}/refined_masks_color.png",refined_masks_color)



color_list = [[127, 123, 229], [195, 240, 251], [120, 200, 255],
               [243, 241, 230], [224, 190, 144], [178, 116, 75],
               [255, 100, 0], [0, 255, 100],
              [100, 0, 255], [100, 255, 0], [255, 0, 255],
              [0, 255, 255], [192, 192, 192], [128, 128, 128],
              [128, 0, 0], [128, 128, 0], [0, 128, 0],
              [128, 0, 128], [0, 128, 128], [0, 0, 128]]
        


class ClassHistogram():
    def __init__(self, image_paths, feature_extractor, config):
        self.image_paths = image_paths
        self.feature_extractor = feature_extractor
        self.config = config
        self.info = ""
        #####################
        # process config
        self.config['mask_path'] = self.config['project_root'] + "/datasets/masks"
        self.config['ckpt_path'] = self.config['project_root'] + "/ckpt"
        self.config['grounding_config']['project_root'] = self.config['project_root']
        self.config['grounding_config']['ckpt_path'] = self.config['ckpt_path']
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.config['com_config'] = {}
        self.config['com_config']['transform'] = transform
        #######################
        self.feature_dims = {
            'area':1,
            'color':1,
            'position':2,
            'scale':1,
            'cnn_shape':2048,
            'cnn_image':2048,
        }
        self.in_dim = np.sum([self.feature_dims[i] for i in self.config['feature_list']])
        self.projector = torch.nn.Linear(self.in_dim,
                                   config["proj_dim"])
        self.projector.bias.data.zero_()
        self.projector.weight.data.normal_(mean=0.0, std=0.01)
        self.projector = self.projector.cuda()
        #######################


        if not os.path.exists(f"{self.config['mask_path']}/{self.config['category']}"):
            os.makedirs(f"{self.config['mask_path']}/{self.config['category']}",exist_ok=True)
            # filter some background and noise
            grounding_segmentation(self.image_paths,f"{self.config['mask_path']}/{self.config['category']}",config['grounding_config'])
            segmentation(self.image_paths,
                f"{self.config['mask_path']}/{self.config['category']}",
                use_grounding_filter=self.config['use_grounding_filter'],
                no_sam=self.config['no_sam'])
            
        self.info_path = f"{self.config['mask_path']}/{self.config['category']}/info"
        os.makedirs(self.info_path,exist_ok=True)

        self.read_all_masks()

        # further filter noise by clustering feature of components

        #########################################
        # load component features
        if os.path.exists(f"{self.info_path}/all_component_feats.npz"):
            print("load component features.")
            self.all_component_feats = np.load(f"{self.info_path}/all_component_feats.npz")
        #if not exist, extract component features
        else:
            com_config = self.config['com_config']
            self.component_feature_extractor = ComponentFeatureExtractor(com_config,model=self.feature_extractor)
            self.extract_component_features()
        #########################################
            
        #########################################
        # cluster component features
        print("clustering component features...")
        self.info += "## Cluster Info ##\n"
        if self.config['no_cluster']:
            self.info += "No cluster\n"
            self.component_labels = np.zeros([len(self.all_component_feats['cnn_image'])]).astype(np.int32)
        else:
            self.cluster_component_features(feature_list=self.config['feature_list'])
        self.save_cluster_map()
        #################################################



        #######################################
        # visualize
        fig,ax = plt.subplots()
        fig.set_dpi(300)
        ax.set_facecolor('black')
        color_list_rgb = np.array(color_list)/255
        # bgr to rgb
        color_list_rgb = color_list_rgb[:,::-1]
        vis_feature = self.all_component_feats.copy()
        # T-SNE
        import sklearn.manifold
        vis_feature = sklearn.manifold.TSNE(n_components=2).fit_transform(vis_feature)
        for cls in range(np.max(self.component_labels)+1):
            plt.plot(vis_feature[self.component_labels==cls,0],vis_feature[self.component_labels==cls,1],'o',markersize=3,c=color_list_rgb[cls])
        #plt.plot(self.all_component_feats[:,0],self.all_component_feats[:,0]*0,'o',markersize=1,alpha=0.1)
        plt.savefig(self.config['mask_path']+"/"+self.config['category']+"/info/feature.png")
        plt.close()
        # def show_anns(img,mask,idx):
        #     masks = split_masks_from_one_mask(mask)
        #     labels = self.component_labels[self.image_com_ids[idx]:self.image_com_ids[idx+1]]
        #     mask_color = np.zeros_like(img)
        #     for i,m in enumerate(masks):
        #         if labels[i] != -1:
        #             mask_color[m!=0] = color_list[labels[i]]
        #     result = np.hstack([img,mask_color])
        #     cv2.imwrite("result.png",result)
        #     cv2.imshow("result",result)
        #     cv2.waitKey(0)
        # #show_anns(self.images[0],self.masks[0],0)
        # for i in range(1,50):
        #     show_anns(self.images[i],self.masks[i],i)
        #######################################

        #######################################
        # build histogram and save
        self.build_histogram()
        self.remap_cluster_map_and_histogram()
        with open(self.info_path+"/info.txt",'w') as f:
            f.write(self.info)
        print("done!")




    def read_all_masks(self,resize_padding=False):
        print("reading images and masks...")
        mask_padding,mask_inv_padding = get_padding_functions(Image.open(self.image_paths[0]).size,self.config['image_size'],mode='nearest')
        padding,padding_inv = get_padding_functions(Image.open(self.image_paths[0]).size,self.config['image_size'])
        
        if self.config['no_sam']:
            # 兼容多种图像格式 - 根据原始图像路径构建掩膜路径
            mask_paths = []
            for img_path in self.image_paths:
                # 移除文件扩展名，构建掩膜路径
                img_path_no_ext = img_path
                for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
                    if img_path_no_ext.endswith(ext):
                        img_path_no_ext = img_path_no_ext[:-len(ext)]
                        break
                mask_name = '/'.join(img_path_no_ext.split("/")[-3:])
                mask_paths.append(f"{self.config['mask_path']}/{mask_name}/grounding_mask.png")
            self.masks = [Image.open(mask_path).convert("L") for mask_path in tqdm.tqdm(mask_paths)]
        else:
            # 兼容多种图像格式 - 根据原始图像路径构建掩膜路径
            mask_paths = []
            for img_path in self.image_paths:
                # 移除文件扩展名，构建掩膜路径
                img_path_no_ext = img_path
                for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
                    if img_path_no_ext.endswith(ext):
                        img_path_no_ext = img_path_no_ext[:-len(ext)]
                        break
                mask_name = '/'.join(img_path_no_ext.split("/")[-3:])
                mask_paths.append(f"{self.config['mask_path']}/{mask_name}/refined_masks.png")
            self.masks = [Image.open(mask_path).convert("L") for mask_path in tqdm.tqdm(mask_paths)]
        self.images = [Image.open(p).convert("RGB") for p in tqdm.tqdm(self.image_paths)]
        
        print("padding images and masks...")
        if resize_padding:
            self.images = [np.array(padding(img)) for img in self.images]
            self.masks = [np.array(mask_padding(m)) for m in self.masks]
        else:
            self.images = [np.array(img) for img in self.images]
            self.masks = [np.array(m) for m in self.masks]

        # for m in split_masks_from_one_mask(self.masks[65]):
        #     cv2.imshow(f"mask{np.sum(m!=0)}",m)
        #     cv2.waitKey(0)
            
        self.image_com_ids = [len(split_masks_from_one_mask(m)) for m in self.masks]
        self.component_num_mode = scipy.stats.mode(self.image_com_ids,keepdims=False).mode
        
        np.save(f"{self.info_path}/component_num_mode.npy",self.component_num_mode)
        self.image_com_ids = np.cumsum(self.image_com_ids)
        self.image_com_ids = np.insert(self.image_com_ids,0,0)

    def extract_component_features(self):
        # gather all component features into one list
        # self.all_component_feats = np.ndarray: [number_of_all_components, feature_dim_of_component]
        self.all_component_feats = {'area':[],'color':[],'position':[],'cnn_shape':[],'cnn_image':[]}
        for image,mask in tqdm.tqdm(zip(self.images,self.masks),desc="extracting component features..."):
            # masks = split_masks_from_one_mask(mask)
            # for m in masks:
            #     cv2.imshow("mask",m)
            #     cv2.waitKey(0)
            features = self.component_feature_extractor.extract(image,
                                                                  split_masks_from_one_mask(mask),
                                                                  ) # [number_of_mask, feature_dim_of_component]
            for feature_name in ['area','color','position','cnn_image']:
                self.all_component_feats[feature_name].append(features[feature_name])

            
        for feature_name in ['area','color','position','cnn_image']:
            self.all_component_feats[feature_name] = np.concatenate(self.all_component_feats[feature_name],axis=0)
        np.savez(f"{self.info_path}/all_component_feats.npz",**self.all_component_feats)


    def cluster_component_features(self,feature_list=['area','color','position','cnn_image']):
        result_feature = list()
        self.info += f"Used feature: {feature_list}\n"
        for feature_name in feature_list:
            result_feature.append(self.all_component_feats[feature_name])
        self.all_component_feats = np.concatenate(result_feature,axis=1)
        if self.config['reduce_dim']:
            self.info += f"Reduce dim from {self.all_component_feats.shape[1]} to {self.config['proj_dim']}\n"
            self.all_component_feats = self.projector(torch.from_numpy(self.all_component_feats).type(torch.float32).cuda())
            self.all_component_feats = self.all_component_feats.detach().cpu().numpy()
            print(f"feature shape:{self.all_component_feats.shape}")
        self.info += f"Feature shape:{self.all_component_feats.shape}\n"

        ###################################
        # mean shift
        t = time.time()
        hyper = self.config['max_hyper']
        self.cluster_model = sklearn.cluster.MeanShift(
                                                        bandwidth=hyper,
                                                        cluster_all=True,
                                                        n_jobs=-1,
                                                    ).fit(self.all_component_feats)

        print(f"mean shift time:{time.time()-t}")
        
        self.component_labels = self.cluster_model.labels_
        # remove small clusters
        for i in range(np.max(self.component_labels)+1):
            class_num = np.sum(self.component_labels==i)
            if class_num < int(len(self.image_paths)*0.5):
                self.component_labels[self.component_labels==i] = -1
        print(f"number of clusters:{np.max(self.component_labels)+1} ,hyper={hyper}")
        self.info += f"number of clusters:{np.max(self.component_labels)+1} ,hyper={hyper}\n"
        print()

    def save_cluster_map(self):
        for i in tqdm.tqdm(range(len(self.image_paths)),desc="saving cluster maps..."):
            mask = self.masks[i]
            masks = split_masks_from_one_mask(mask)

            labels = self.component_labels[self.image_com_ids[i]:self.image_com_ids[i+1]]

            mask_cluster = np.zeros_like(mask)
            mask_cluster_color = np.zeros([mask.shape[0],mask.shape[1],3],dtype=np.uint8)
            for j,m in enumerate(masks):
                if labels[j] != -1:
                    mask_cluster[m!=0] = labels[j]+1
                    mask_cluster_color[m!=0] = color_list[labels[j]]
            
            # 兼容多种图像格式 - 根据原始图像路径构建保存路径
            img_path = self.image_paths[i]
            img_path_no_ext = img_path
            for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
                if img_path_no_ext.endswith(ext):
                    img_path_no_ext = img_path_no_ext[:-len(ext)]
                    break
            mask_name = '/'.join(img_path_no_ext.split("/")[-3:])
            save_dir = f"{self.config['mask_path']}/{mask_name}"
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f"{save_dir}/cluster_map.png",mask_cluster)
            cv2.imwrite(f"{save_dir}/cluster_map_color.png",mask_cluster_color)

    
    def build_histogram(self):
        histograms = list()
        for i in tqdm.tqdm(range(len(self.image_paths)),desc="building histogram..."):
            mask = self.masks[i]
            masks = split_masks_from_one_mask(mask)
            labels = self.component_labels[self.image_com_ids[i]:self.image_com_ids[i+1]]
            histogram = np.zeros([np.max(self.component_labels)+1]) 
            for j,m in enumerate(masks):
                if labels[j] != -1:
                    histogram[labels[j]] += np.sum(m!=0)
            #histogram[-1] = np.sum(np.ones_like(masks[0])) - np.sum(histogram[:-1])
            histogram = histogram / (mask.shape[0] * mask.shape[1])
            fig,ax = plt.subplots()
            ax.set_facecolor('black')
            color_list_rgb = np.array(color_list)/255
            # bgr to rgb
            color_list_rgb = color_list_rgb[:,::-1]
            for j in range(len(histogram)):
                plt.bar(j,histogram[j],color=color_list_rgb[j])
            # 兼容多种图像格式 - 根据原始图像路径构建保存路径
            img_path = self.image_paths[i]
            img_path_no_ext = img_path
            for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
                if img_path_no_ext.endswith(ext):
                    img_path_no_ext = img_path_no_ext[:-len(ext)]
                    break
            mask_name = '/'.join(img_path_no_ext.split("/")[-3:])
            save_dir = f"{self.config['mask_path']}/{mask_name}"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/histogram.png")
            plt.close()
            histograms.append(histogram)
        histograms = np.array(histograms)

        # filter out some images
        cluster = sklearn.cluster.HDBSCAN(
                                    min_cluster_size=int(len(self.image_paths)*0.1)
                                    ).fit(histograms)
        s=1
        while np.max(cluster.labels_) == -1:
            cluster = sklearn.cluster.HDBSCAN(
                                    min_cluster_size=int(len(self.image_paths)*0.1)-s
                                    ).fit(histograms)
            s+=1
        print(f"number of hist clusters:{np.max(cluster.labels_)+1}")
        self.info += f"\n\n## Histograms ##\n"
        self.info += f"number of hist clusters:{np.max(cluster.labels_)+1}\n"
        # pick the largest cluster
        cluster_num = np.max(cluster.labels_)+1
        cluster_size = np.zeros([cluster_num])
        for i in range(cluster_num):
            cluster_size[i] = np.sum(cluster.labels_==i)
            #print(f"hist cluster {i} has {cluster_size[i]} images")
        max_cluster = np.argmax(cluster_size)
        self.info += f"max cluster:{max_cluster}\n"
        indices = np.where(cluster.labels_==max_cluster)[0]
        ################################
        #filter each cluster with good(close to mean) histogram
        topk_ratio = 0.5
        self.info += f"\n\n## Filtered Histograms ##\n"
        self.info += f"topk_ratio:{topk_ratio}\n"
        good_indices = list()
        for i in range(cluster_num):
            mean_hist = np.mean(histograms[cluster.labels_==i],axis=0)
            fig,ax = plt.subplots()
            ax.set_facecolor('black')
            color_list_rgb = np.array(color_list)/255
            # bgr to rgb
            color_list_rgb = color_list_rgb[:,::-1]
            for j in range(len(mean_hist)):
                plt.bar(j,mean_hist[j],color=color_list_rgb[j])
            plt.savefig(f"{self.info_path}/mean_hist_{i}.png")
            plt.close()
            num_good = int(len(histograms[cluster.labels_==i])*topk_ratio)
            max_dist = np.max(np.abs((histograms[cluster.labels_==i]-mean_hist)),axis=1)
            sum_dist = np.sum(np.abs((histograms[cluster.labels_==i]-mean_hist)),axis=1)
            good_indices.append(np.argsort(np.max(np.abs((histograms-mean_hist)),axis=1))[:num_good])
            print(f"hist cluster {i} filtered from {len(histograms[cluster.labels_==i])} to {len(good_indices[i])}")
        if self.config['filter_hist']:
            good_indices = good_indices[max_cluster]
        else:
            good_indices = np.concatenate(good_indices)
        good_indices = np.sort(good_indices)
        self.info += f"filter to one cluster:{self.config['filter_hist']}\n"
        self.info += f"number of good histogram:{len(good_indices)}\n"

        print(f"number of good histogram:{len(good_indices)}")
        self.info += f"{good_indices}\n"
        print(good_indices)

        ################################
        # filter the labels which area is too small
        self.info += f"\n\n## Filtered Labels ##\n"
        self.info += f"area threshold: 0.001\n"
        thresh = 0.001
        mean_good_hist = np.mean(histograms[indices],axis=0)
        self.filtered_labels = np.where(mean_good_hist>thresh)[0]
        self.info += f"filter from {list(range(mean_good_hist.shape[0]))} to {self.filtered_labels.tolist()}\n"
        self.label_map = {i:j for i,j in zip(self.filtered_labels,range(len(self.filtered_labels)))}
        np.save(f"{self.info_path}/raw_component_histogram.npy",np.array(histograms))
        np.save(f"{self.info_path}/filtered_histogram_indices.npy",good_indices)
    
    def remap_cluster_map_and_histogram(self):
        for i in tqdm.tqdm(range(len(self.image_paths)),desc="remapping cluster map and histogram..."):
            mask = self.masks[i]
            masks = split_masks_from_one_mask(mask)

            labels = self.component_labels[self.image_com_ids[i]:self.image_com_ids[i+1]]

            mask_cluster = np.zeros_like(mask)
            mask_cluster_color = np.zeros([mask.shape[0],mask.shape[1],3],dtype=np.uint8)
            for j,m in enumerate(masks):
                if labels[j] != -1 and labels[j] in self.filtered_labels:
                    mask_cluster[m!=0] = self.label_map[labels[j]]+1
                    mask_cluster_color[m!=0] = color_list[self.label_map[labels[j]]]
            # 兼容多种图像格式 - 根据原始图像路径构建保存路径
            img_path = self.image_paths[i]
            img_path_no_ext = img_path
            for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
                if img_path_no_ext.endswith(ext):
                    img_path_no_ext = img_path_no_ext[:-len(ext)]
                    break
            mask_name = '/'.join(img_path_no_ext.split("/")[-3:])
            save_dir = f"{self.config['mask_path']}/{mask_name}"
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f"{save_dir}/filtered_cluster_map.png",mask_cluster)
            cv2.imwrite(f"{save_dir}/filtered_cluster_map_color.png",mask_cluster_color)

        histograms = list()
        for i in tqdm.tqdm(range(len(self.image_paths)),desc="building histogram..."):
            mask = self.masks[i]
            masks = split_masks_from_one_mask(mask)
            labels = self.component_labels[self.image_com_ids[i]:self.image_com_ids[i+1]]
            histogram = np.zeros([len(self.filtered_labels)]) 
            for j,m in enumerate(masks):
                if labels[j] != -1 and labels[j] in self.filtered_labels:
                    histogram[self.label_map[labels[j]]] += np.sum(m!=0)
            #histogram[-1] = np.sum(np.ones_like(masks[0])) - np.sum(histogram[:-1])
            histogram = histogram / (mask.shape[0] * mask.shape[1])
            fig,ax = plt.subplots()
            ax.set_facecolor('black')
            color_list_rgb = np.array(color_list)/255
            # bgr to rgb
            color_list_rgb = color_list_rgb[:,::-1]
            for j in range(len(histogram)):
                plt.bar(j,histogram[j],color=color_list_rgb[j])
            # 兼容多种图像格式 - 根据原始图像路径构建保存路径
            img_path = self.image_paths[i]
            img_path_no_ext = img_path
            for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
                if img_path_no_ext.endswith(ext):
                    img_path_no_ext = img_path_no_ext[:-len(ext)]
                    break
            mask_name = '/'.join(img_path_no_ext.split("/")[-3:])
            save_dir = f"{self.config['mask_path']}/{mask_name}"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/filtered_histogram.png")
            plt.close()
            histograms.append(histogram)
        histograms = np.array(histograms)
        np.save(f"{self.info_path}/filtered_component_histogram.npy",np.array(histograms))

if __name__ == "__main__":
    import timm
    from torchvision import transforms
    categories = ['breakfast_box','juice_bottle','pushpins','screw_bag','splicing_connectors',]
    project_root = os.path.dirname(os.path.abspath(__file__))
    for category in categories:
        
        config = read_config(f"{project_root}/configs/class_histogram/{category}.yaml")
        config['project_root'] = project_root

        image_paths = glob.glob(f"{project_root}/datasets/mvtec_loco/{config['category']}/train/good/*.png")
    
        feature_extractor = timm.create_model('wide_resnet50_2'
                                            ,pretrained=True,
                                            features_only=True,
                                            out_indices=[4])
        feature_extractor.eval()
        feature_extractor.cuda()
        ch = ClassHistogram(image_paths,feature_extractor=feature_extractor,config=config)
    
