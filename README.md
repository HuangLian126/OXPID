# OXPID

This repository is based on the **[OpenDet](https://github.com/csuhan/opendet2)** codebase.  

## 1. Environment Requirements

We recommend using the same environment settings as OpenDet.  
Below is a tested configuration:

- Python 3.7+
- PyTorch 1.9.1
- CUDA 11.3
- Detectron2 0.6
- torchvision 0.9.1
- Other dependencies: `yacs`, `tqdm`, `scipy`, `opencv-python`
git clone https://github.com/csuhan/opendet2.git
cd opendet2
## 2. Setup

```bash
# 1. Download the OpenDet code from the official repository
git clone https://github.com/csuhan/opendet2.git
cd opendet2

# 2. Replace the original fast_rcnn.py file with the one provided in this repository
cp ../OXPID/fast_rcnn.py detectron2/modeling/roi_heads/fast_rcnn.py

# 3. Copy the PNPL and D2D modules into the appropriate directory
cp ../OXPID/PNPL.py detectron2/modeling/roi_heads/
cp ../OXPID/D2D.py  detectron2/modeling/roi_heads/
