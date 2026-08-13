# OXPID

This repository is built upon the **[OpenDet](https://github.com/csuhan/opendet2)** codebase.  
We provide the implementation of the proposed **PNPL** and **U²D** modules for open-set X-ray prohibited item detection.

📂 **The first OXPID Benchmark:** [Download Link](https://1drv.ms/u/c/4d26ab976d8445b4/IQD54S0e0EVDS55gd1C4T2yPAbkFDkVKPBc6JCu6F2pJ0FI?e=jxnLTG)

📂 **The second OXPID Benchmark:** [Download Link](https://pan.baidu.com/s/1r6Ornktr0gva_Kg7Qr5b8g) code:nma1

## ⚙️ Step 1. Environment Setup

We recommend using the same environment settings as OpenDet. The following configuration has been tested:

-  Python 3.7+
-  PyTorch 1.9.1
-  CUDA 11.3
-  Detectron2 0.6
-  torchvision 0.9.1
-  Other dependencies: `yacs`, `tqdm`, `scipy`, `opencv-python`

## 📥 Step 2. Download and Setup

Clone the OpenDet repository:

```bash id="3gmnyh"
git clone https://github.com/csuhan/opendet2.git
cd opendet2
```

Replace the original `fast_rcnn.py` with the modified version provided in this repository:

```bash id="xbjzcg"
cp ../OXPID/fast_rcnn.py detectron2/modeling/roi_heads/fast_rcnn.py
```

## 🚀 Step 3. Training

Train the OXPID model using a single GPU:

```bash id="el98b6"
python tools/train_net.py \
    --num-gpus 1 \
    --config-file configs/faster_rcnn_R_50_FPN_3x_opendet.yaml
```
