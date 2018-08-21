# PSPNet-Pytorch
This Repo contains an implemetation of "Pyramid Scene Parsing Network" in Pytorch. Pretrained weights are converted from Official Caffe Repo. The performance is almost 76% mean-IoU on Validation set of CityScapes Dataset.

## Installation
1. Install Anaconda3 from [Here](https://www.anaconda.com)
2. Create a Conda Environment: conda env create -f environment.yml
3. Download Cityscapes Dataset sequence data from [Here](https://www.cityscapes-dataset.com/)
4. Download Caffe Pretrained from [Here](https://drive.google.com/open?id=0BzaU285cX7TCT1M3TmNfNjlUeEU) and put it in Caffe-PSPNet folder
4. Run python main.py

## Note

Weight convesion from Caffe to Pytorch is modified from [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).
Preprocessing and loss function is modified from [pytorch-semantic-segmentation](https://github.com/zijundeng/pytorch-semantic-segmentation)
Since the differences were huge, I decided not to add a branch to any of the the above project.
