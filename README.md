# UW-NIWA-tracking

The code will be consists three part (will be organize and upload in the next few days):
1) Detection (based on [mmyolo](https://github.com/open-mmlab/mmyolo))
2) Tracking (based on [mmtracking](https://github.com/open-mmlab/mmtrack))
3) Classification (based on [mmcls](https://github.com/open-mmlab/mmpretrain/tree/v0.17.0))

All of them are built on top of [open-mmlab](https://github.com/open-mmlab). All experiments including training and infencing are done on a Nvidia Quadro GV100 using Python 3.7 and `PyTorch==1.6.0` with `cudatoolkit==10.2`.

MMYOLO relies on PyTorch, MMCV, MMEngine, and MMDetection. Below are quick steps for installation. Please refer to the [Install Guide](docs/en/get_started/installation.md) for more detailed instructions.

## Recommendated Installation 

```shell
conda create -n open-mmlab python=3.7 pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc6,<3.1.0"
# Install mmyolo
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install mmyolo
mim install -v -e .
cd ..
# Install mmtrack
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
pip install git+https://github.com/JonathonLuiten/TrackEval.git
cd ..
# Install mmcls
git clone https://github.com/open-mmlab/mmpretrain/tree/v0.17.0
cd mmclassification
pip install -e .  # or "python setup.py develop"
```
