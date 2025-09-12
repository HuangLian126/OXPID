# OXPID

This repository is based on the **[OpenDet](https://github.com/csuhan/opendet2)** codebase.  

## 1. Environment Requirements

We recommend using the same environment settings as OpenDet.  
Below is a tested configuration:

- Python 3.8+
- PyTorch 1.8.1
- CUDA 10.2 / 11.x
- Detectron2 0.4 (compatible with PyTorch 1.8)
- torchvision 0.9.1
- Other dependencies: `yacs`, `tqdm`, `scipy`, `opencv-python`

You can install dependencies with:

```bash
conda create -n oxpid python=3.8 -y
conda activate oxpid

# install PyTorch + CUDA
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# install Detectron2
pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

# other packages
pip install yacs tqdm scipy opencv-python
