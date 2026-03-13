# MedAD: A Training-free Unified Model for Medical Anomaly Detection

Official implementation of paper [MedAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection]

---
## Introduction

MedAD is a **training-free** unified model for few-shot medical anomaly detection. It can detect anomalies across various medical imaging modalities using only a few normal reference samples, without requiring domain-specific training.


![MedAD Architecture](https://github.com/user-attachments/files/25932678/1.pdf)


---

## Key Features

- **Training-free**: No need for extensive training on target datasets
- **Few-shot Learning**: Works with only 1-2 normal reference samples
- **Multi-modal**: Supports various medical imaging modalities
- **State-of-the-art**: Achieves superior performance on 7 medical datasets
- **Unified Model**: Single model for multiple medical imaging tasks

---

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/labcbb/MedAD.git
cd MedAD
```

### Step 2: Create Conda Environment

We recommend using Python 3.9-3.11:

```bash
# Create a new conda environment
conda create -n medad python=3.11 -y

# Activate the environment
conda activate medad
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (CUDA 11.8 example, adjust based on your system)
conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt

# Install GroundingDINO
cd models/GroundingDINO
pip install -e .
cd ../..
```

**GPU Requirements**: A GPU with at least 16GB memory is recommended (e.g., NVIDIA RTX 3090, V100, A100).

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

---

## Download Pretrained Models

MedAD relies on several pretrained vision foundation models. Download them to the `pretrained_ckpts/` folder.

### Automated Download Script

```bash
# Navigate to pretrained_ckpts folder
cd pretrained_ckpts

# Download SAM-HQ model
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth

# Download GroundingDINO model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download CLIP ViT-L/14@336px
wget https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt -O Clip-ViT-L-14-336px.pt

# Download CLIP ViT-B/32
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

# Download DINOv2 ViT-g/14
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth

# Download DINO ViT-S/8
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth

# Download MedSAM (optional, for medical image segmentation)
wget https://drive.google.com/uc?id=1UAmWL88roYR7wKlnApw5Bcgo7GCQO4xK -O medsam_vit_b.pth

# Return to project root
cd ..
```

### Manual Download

If `wget` fails, manually download from the links above and place them in `pretrained_ckpts/`.

### Verify Downloads

```bash
ls -lh pretrained_ckpts/
```

Expected output:
```
Clip-ViT-L-14-336px.pt          (~890 MB)
dino_deitsmall8_300ep_pretrain.pth    (~83 MB)
dinov2_vitg14_pretrain.pth      (~4.2 GB)
groundingdino_swint_ogc.pth     (~662 MB)
medsam_vit_b.pth                (~358 MB, optional)
sam_hq_vit_h.pth                (~2.4 GB)
ViT-B-32.pt                     (~338 MB)
```

---

## Prepare Medical Datasets

MedAD is evaluated on 7 medical imaging datasets spanning 6 different imaging modalities.

### Download Medical Datasets

#### Download from Zenodo

```bash
# Navigate to data folder
cd data

# Download medical datasets
wget https://zenodo.org/records/18981810/files/BUSI.tar.gz?download=1 -O BUSI.tar.gz

# Extract datasets
tar -xzf BUSI.tar.gz

# Verify extraction
ls -d BraTS21 BUSI RESC ISIC18 HIS OCT2017

cd ..

### Dataset Directory Structure

After extraction, your `data/` folder should look like this:

```
data/
├── BrainMRI/
│   ├── train/
│   │   └── good/          # Normal training samples
│   ├── test/
│   │   ├── good/          # Normal test samples
│   │   └── anomaly/       # Anomalous test samples
│   └── ground_truth/
│       └── anomaly/       # Pixel-level masks
├── RESC/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   └── anomaly/
│   └── ground_truth/
│       └── anomaly/
├── HIS/
│   ├── train/
│   │   └── good/
│   └── test/
│       ├── good/
│       └── anomaly/
└── OCT2017/
    ├── train/
    │   └── good/
    └── test/
        ├── good/
        └── anomaly/
```

### Generate Metadata Files

Generate `meta.json` for each dataset to index all images:

```bash
# Run the metadata generation script
python data/medical_slover.py
```

This will create `meta.json` files in each dataset folder containing paths to all images and annotations.

---

## Run MedAD

### Step 1: Component Segmentation (Preprocessing)

Perform contextual component clustering for all datasets. This step segments anatomical components in advance to speed up inference.

```bash
python segment_components.py
```

**Note**: This step may take several hours depending on your dataset size and GPU. You only need to run it once.

Expected output:
```
Processing BraTS21...
Processing BUSI...
Processing RESC...
Processing ISIC18...
Processing HIS...
Processing OCT2017...
Component segmentation completed!
```

### Step 2: Run Anomaly Detection

#### Test on All Medical Datasets

```bash
bash test_medical.sh
```

#### Test on Individual Datasets

**Datasets with pixel-level annotations:**

```bash
# BraTS
python test_medad.py --dataset brainmri --data_path ./data/BrainMRI --round 0 --image_size 224 --k_shot 1

# RESC
python test_medad.py --dataset resc --data_path ./data/RESC --round 0 --image_size 224 --k_shot 1
```

**Datasets with image-level annotations only:**

```bash
# HIS
python test_medad_no_pixel.py --dataset his --data_path ./data/HIS --round 0 --image_size 224 --k_shot 1

# OCT2017
python test_medad_no_pixel.py --dataset oct2017 --data_path ./data/OCT2017 --round 0 --image_size 224 --k_shot 1
```

### Step 3: View Results

Results will be saved in the `results/` folder:

```bash
ls results/
```

Each dataset will have a subfolder containing:
- `metrics.json`: Image-AUROC, Pixel-AUROC
- `predictions/`: Anomaly maps for each test image
- `visualizations/`: Side-by-side comparisons (original, ground truth, prediction)

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (brainmri, liverct, resc, chestxray, his, oct17) | Required |
| `--data_path` | Path to dataset folder | Required |
| `--k_shot` | Number of reference samples (1, 2, 4) | 1 |
| `--image_size` | Input image size | 224 |
| `--round` | Random seed for reference sample selection | 0 |
| `--save_vis` | Save visualization results | False |
| `--vis_num` | Maximum number of visualizations to save | 100 |

### Example: 2-shot Learning

```bash
python test_medad.py --dataset brainmri --data_path ./data/BrainMRI --k_shot 2 --round 0 --image_size 224
```

### Example: Save Visualizations

```bash
python test_medad.py --dataset brainmri --data_path ./data/BrainMRI --k_shot 1 --round 0 --save_vis --vis_num 50
```

---


## Expected Performance

Performance on medical datasets (1-shot setting):

| Dataset | Image-AUROC | Pixel-AUROC |
|---------|-------------|-------------|
| BraTS21 | 95.3% | 95.7% |
| BUSI | 82.9% | 84.6% |
| RESC | 84.6% | 94.0% |
| ISIC18 | 98.7% | 85.1% |
| HIS | 72.1% | - |
| OCT2017 | 88.5% | - |


---

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

```bash
# Reduce image size
python test_medad.py --dataset brainmri --data_path ./data/BraTS21 --image_size 192 --k_shot 1

# Or use a smaller CLIP model (edit MedAD.py to use ViT-B/32 instead of ViT-L/14)
```

### Missing Pretrained Models

Ensure all models are downloaded:

```bash
ls pretrained_ckpts/
# Should show: Clip-ViT-L-14-336px.pt, sam_hq_vit_h.pth, groundingdino_swint_ogc.pth, etc.
```

### Import Errors

```bash
# Reinstall GroundingDINO
cd models/GroundingDINO
pip install -e .
cd ../..

# Check if all dependencies are installed
pip list | grep -E "torch|clip|opencv|pillow"
```

### Slow Inference

- **First run is slow**: Component segmentation caches results; subsequent runs are faster
- **Run preprocessing first**: `python segment_components.py`
- **Use GPU**: Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`


**Happy detecting!**
