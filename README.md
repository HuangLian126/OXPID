# OXPID

# ==========================================================
# 1. Environment Setup
# ==========================================================

# Create and activate conda environment
conda create -n oxpid python=3.8 -y
conda activate oxpid

# Install PyTorch + CUDA (PyTorch 1.8.1 + CUDA 10.2)
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install Detectron2 (compatible with PyTorch 1.8.1)
pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

# Install other dependencies
pip install yacs tqdm scipy opencv-python


# ==========================================================
# 2. Download and Setup Code
# ==========================================================

# Download the OpenDet code
git clone https://github.com/csuhan/opendet2.git
cd opendet2

# Replace the original fast_rcnn.py file with the one provided in this repository
cp ../OXPID/fast_rcnn.py detectron2/modeling/roi_heads/fast_rcnn.py

# Copy the PNPL and D2D modules into the appropriate directory
cp ../OXPID/PNPL.py detectron2/modeling/roi_heads/
cp ../OXPID/D2D.py  detectron2/modeling/roi_heads/


# ==========================================================
# 3. Training
# ==========================================================

# Train the OXPID model with 4 GPUs
python tools/train_net.py \
  --num-gpus 4 \
  --config-file configs/opendet/oxpid_R_50_C4.yaml \
  OUTPUT_DIR ./output/oxpid_R_50_C4


# ==========================================================
# 4. Evaluation
# ==========================================================

# Evaluate a trained OXPID model
python tools/train_net.py \
  --num-gpus 4 \
  --config-file configs/opendet/oxpid_R_50_C4.yaml \
  --eval-only \
  MODEL.WEIGHTS ./output/oxpid_R_50_C4/model_final.pth

