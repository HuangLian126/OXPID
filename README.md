# OXPID

This repository is based on the **[OpenDet](https://github.com/csuhan/opendet2)** codebase.  
We provide modifications that integrate the **PNPL** and **D2D** modules to improve open-set X-ray prohibited item detection.

📂 **OXPID Dataset**: [Download Link](https://1drv.ms/u/c/4d26ab976d8445b4/EXJgOsVEydpChdT-0Y5qbTEB2NeAhR2jjY5F0BwlM1wG0A?e=XfLDst)

```bash
############################################################
# Step 1. Environment Setup
############################################################

We recommend using the same environment settings as OpenDet. Below is a tested configuration:
- Python 3.7+
- PyTorch 1.9.1
- CUDA 11.3
- Detectron2 0.6
- torchvision 0.9.1
- Other dependencies: `yacs`, `tqdm`, `scipy`, `opencv-python`

############################################################
# Step 2. Download and Setup Code
############################################################

# Download the OpenDet code
git clone https://github.com/csuhan/opendet2.git
cd opendet2

# Replace the original fast_rcnn.py file with the one provided in this repository
cp ../OXPID/fast_rcnn.py detectron2/modeling/roi_heads/fast_rcnn.py

############################################################
# Step 3. Training
############################################################

# Train the OXPID model with 1 GPUs
python tools/train_net.py --num-gpus 1 --config-file configs/faster_rcnn_R_50_FPN_3x_opendet.yaml
############################################################
# Ste
