# GEPT-Net: An efficient geometry enhanced point transformer for shield tunnel leakage segmentation
<div align="center">
Jundi Jiang</a><sup>1</sup>, Yueqian Shen</a><sup>1</sup>, Jinhu Wang</a><sup>2</sup>, Jinguo Wang</a><sup>1</sup>， Chenyang Zhang</a><sup>1</sup>, Jingyi Wang</a><sup>1</sup>, Vagner Ferreira</a><sup>1</sup>

<div align="center">
</a><sup>1</sup>School of Earth Sciences and Engineering，Hohai University， </a><sup>2</sup>Institute for Biodiversity and Ecosystem Dynamics (IBED)， University of Amsterdam

![teaser](./doc/GEPT-Net.png)

<div align="left">

# Notes 	
<div align="left">
  
We adapt the codebase of [Pointcept](https://github.com/Pointcept/Pointcept) which  is a powerful and flexible codebase for point cloud perception research. Please refer to [Pointcept](https://github.com/Pointcept/Pointcept) if you need more information.

## Installation
We recommend using [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.anaconda.com/miniconda/) to build your virtual environment.

Please follow the [Installation](https://github.com/Pointcept/Pointcept/tree/main#installation) to build your conda environment, or you can refer to [requirements](./requirements.txt).

## Usage
Step 1: activate your environment
```
conda activate your_env_name
```
Step2: get into your project root
```
cd your/project/path
```
Step 3: Change your dataset path in [config.py](https://github.com/jdjiang312/GEPT-Net/blob/3f6f1d7c671f322ccce63de34dccf6864c6a8760/config.py#L71)

Step4: Setting training parameters in in [config.py](https://github.com/jdjiang312/GEPT-Net/blob/3f6f1d7c671f322ccce63de34dccf6864c6a8760/config.py)

Step5: Setting Python environment variables
```
export PYTHONPATH=./
```
Step5: run your projecy
```
python tools/train.py --config-file config.py --options save_path=results/test
```

## Citation
If you find this work useful for your research, please cite our paper:
