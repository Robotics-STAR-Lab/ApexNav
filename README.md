<div align="center">
    <img src="assets/apexnav_logo.jpg" alt="ApexNav Logo" width="200">
    <h2>An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion</h2>
    <strong>
      <em>IEEE Robotics and Automation Letters</em>
    </strong>
    <br>
        <a href="https://zager-zhang.github.io" target="_blank">Mingjie Zhang</a><sup>1</sup>,
        <a href="https://personal.hkust-gz.edu.cn/junma/people-page.html" target="_blank">Yuheng Du</a><sup>1</sup>,
        <a href="https://chengkaiwu.me" target="_blank">Chengkai Wu</a><sup>1</sup>,
        Jinni Zhou<sup>1</sup>,
        Zhenchao Qi<sup>1</sup>,
        <a href="https://personal.hkust-gz.edu.cn/junma/people-page.html" target="_blank">Jun Ma</a><sup>1</sup>,
        <a href="https://robotics-star.com/people" target="_blank">Boyu Zhou</a><sup>2,†</sup>
        <p>
        <h45>
            <sup>1</sup> The Hong Kong University of Science and Technology (Guangzhou). &nbsp;&nbsp;
            <br>
            <sup>2</sup> Southern University of Science and Technology. &nbsp;&nbsp;
            <br>
        </h45>
        <sup>†</sup>Corresponding Authors
    </p>
    <a href="https://arxiv.org/abs/2504.14478"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-red"/></a>
    <a href='https://robotics-star.com/ApexNav'><img src='https://img.shields.io/badge/Project_Page-ApexNav-green' alt='Project Page'></a>

<br>
<br>

<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="assets/video_plant.gif" alt="apexnav_demo" width="80%">
  </a>
  <br>
  <em>
    <!-- TODO -->
    <!-- ApexNav -->
  </em>
</p>

</div>

## 📢 News

- **[22/08/2025]**: Release the main algorithm of ApexNav.
- **[18/08/2025]**: ApexNav is conditionally accepted to RA-L 2025.


## 📜 Introduction

**[RA-L'25]** This repository maintains the implementation of "ApexNav: An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion".

The pipeline of ApexNav is detailed in the overview below.

<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="assets/pipeline.jpg" alt="pipeline" width="80%">
  </a>
</p>

## 🛠️ Installation
> Tested on Ubuntu 20.04 with ROS Noetic and Python 3.9

### 1. Prerequisites

#### 1.1 System Dependencies
``` bash
sudo apt update
sudo apt-get install libarmadillo-dev libompl-dev
```

#### 1.2 FTXUI
A simple cross-platform C++ library for terminal based user interfaces.
``` bash
git clone https://github.com/ArthurSonzogni/FTXUI
cd FTXUI
mkdir build && cd build
cmake ..
make -j
sudo make install
```

#### 1.3 LLM (Optional)
> You can skip LLM configuration and directly use our pre-generated LLM output results in `llm/answers`

ollama 
``` bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
```

#### 1.4 External Code Dependencies
```bash
git clone git@github.com:WongKinYiu/yolov7.git # yolov7
git clone https://github.com/IDEA-Research/GroundingDINO.git # GroundingDINO
```

#### 1.5 Model Weights Download

Download the following model weights and place them in the `data/` directory:
- `mobile_sam.pt`: https://github.com/ChaoningZhang/MobileSAM/tree/master/weights/mobile_sam.pt
- `groundingdino_swint_ogc.pth`: 
  ```bash
  wget -O data/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  ```
- `yolov7-e6e.pt`: 
  ```bash
  wget -O data/yolov7-e6e.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
  ```


### 2. Setup Python Environment

#### 2.1 Clone Repository
``` bash
git clone git@github.com:Robotics-STAR-Lab/ApexNav.git
cd ApexNav
```

#### 2.2 Create Conda Environment
``` bash
conda env create -f apexnav_environment.yaml -y
conda activate apexnav
```

#### 2.3 Pytorch
``` bash
# You can use 'nvcc --version' to check your CUDA version.
# CUDA 11.8
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

#### 2.4 Habitat Simulator
> We recommend using habitat-lab v0.3.1
``` bash
# habitat-lab v0.3.1
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout tags/v0.3.1;
pip install -e habitat-lab

# habitat-baselines v0.3.1
pip install -e habitat-baselines
```

**Note:** Any numpy-related errors will not affect subsequent operations, as long as `numpy==1.23.5` and `numba==0.60.0` are correctly installed.

#### 2.5 Others
``` bash
pip install salesforce-lavis==1.0.2 # -i https://pypi.tuna.tsinghua.edu.cn/simple
cd .. # Return to ApexNav directory
pip install -e .
```

**Note:** Any numpy-related errors will not affect subsequent operations, as long as `numpy==1.23.5` and `numba==0.60.0` are correctly installed.

## 📥 Datasets Download
> Official Reference: https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md

### 🏠 Scene Datasets
Coming soon.

### 🎯 Task Datasets
``` bash
# Create necessary directory structure
mkdir -p data/datasets/objectnav/hm3d
mkdir -p data/datasets/objectnav/mp3d

# HM3D-v0.1
wget -O data/datasets/objectnav/hm3d/v1.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
unzip data/datasets/objectnav/hm3d/v1.zip -d data/datasets/objectnav/hm3d && mv data/datasets/objectnav/hm3d/objectnav_hm3d_v1 data/datasets/objectnav/hm3d/v1 && rm data/datasets/objectnav/hm3d/v1.zip

# HM3D-v0.2
wget -O data/datasets/objectnav/hm3d/v2.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
unzip data/datasets/objectnav/hm3d/v2.zip -d data/datasets/objectnav/hm3d && mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2 && rm data/datasets/objectnav/hm3d/v2.zip

# MP3D
wget -O data/datasets/objectnav/mp3d/v1.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip
unzip data/datasets/objectnav/mp3d/v1.zip -d data/datasets/objectnav/mp3d/v1 && rm data/datasets/objectnav/mp3d/v1.zip
```

Your final folder `data` structure should look like this:
```
data
├── datasets
│   └── objectnav
│       ├── hm3d
│       │   ├── v1
│       │   │   ├── train
│       │   │   ├── val
│       │   │   └── val_mini
│       │   └── v2
│       │       ├── train
│       │       ├── val
│       │       └── val_mini
│       └── mp3d
│           └── v1
│               ├── train
│               ├── val
│               └── val_mini
├── scene_datasets
│   ├── hm3d
│   │   └── val
│   │       ├── 00800-TEEsavR23oF
│   │       ├── 00801-HaxA7YrQdEC
│   │       ├── .....
│   ├── hm3d_v0.2 -> hm3d
│   └── mp3d
│       ├── 17DRP5sb8fy
│       ├── 1LXtFkjw3qL
│       ├── .....
├── groundingdino_swint_ogc.pth
├── mobile_sam.pt
└── yolov7-e6e.pt
```
Note that `train` and `val_mini` are not required and you can choose to delete them.
## 🚀 Usage
> All following commands should be run in the `apexnav` conda environment
### ROS Compilation
``` bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```
### Run VLMs Servers
Each command should be run in a separate terminal.
``` bash
python -m vlm.detector.grounding_dino --port 12181
python -m vlm.itm.blip2itm --port 12182
python -m vlm.segmentor.sam --port 12183
python -m vlm.detector.yolov7 --port 12184
```
### Launch Visualization and Main Algorithm
```bash
source ./devel/setup.bash && roslaunch exploration_manager rviz.launch # RViz visualization
source ./devel/setup.bash && roslaunch exploration_manager exploration.launch # ApexNav main algorithm
```

### 📊 Evaluate Datasets in Habitat
You can evaluate on all episodes of a dataset.
```bash
# Need to source the workspace
source ./devel/setup.bash

# Choose one datasets to evaluate
python habitat_evaluation.py --dataset hm3dv1
python habitat_evaluation.py --dataset hm3dv2 # default
python habitat_evaluation.py --dataset mp3d

# You can also evaluate on one specific episode.
python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10 # episode_id 10
```
If you want to generate evaluation videos for each episode (videos will be categorized by task results), you can use the following command:
```bash
python habitat_evaluation.py --dataset hm3dv2 need_video=true
```

### 🎮 Keyboard Control in Habitat
You can also choose to manually control the agent in the Habitat simulator:
```bash
# Need to source the workspace
source ./devel/setup.bash

python habitat_manual_control.py --dataset hm3dv1 # Default episode_id = 0
python habitat_manual_control.py --dataset hm3dv1 test_epi_num=10 # episode_id = 10
```
## 📋 TODO List

- [x] Release the main algorithm of ApexNav
- [x] Complete Installation and Usage documentation
- [ ] Add datasets download documentation
- [ ] Add acknowledgment documentation
- [ ] Add utility tools documentation
- [ ] Release the code of real-world deployment
- [ ] Add ROS2 support

## ✒️ Citation

```bibtex
@article{zhang2025apexnav,
  title={ApexNav: An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion},
  author={Zhang, Mingjie and Du, Yuheng and Wu, Chengkai and Zhou, Jinni and Qi, Zhenchao and Ma, Jun and Zhou, Boyu},
  journal={arXiv preprint arXiv:2504.14478},
  year={2025}
}
```