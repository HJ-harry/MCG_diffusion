#!/bin/bash

# 1. environment setting
conda create -n MCG python=3.7
conda activate MCG
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt


# 2. model checkpoints
CHECKPOINT_DIR=./checkpoints
mkdir -p "$CHECKPOINT_DIR"
wget -O "$CHECKPOINT_DIR/ffhq_10m.pt" https://www.dropbox.com/s/4r8r6o2n1pumzmg/ffhq_10m.pt?dl=0
wget -O "$CHECKPOINT_DIR/imagenet256.pt" https://www.dropbox.com/s/rtit2qsb353262t/imagenet256.pt?dl=0
wget -O "$CHECKPOINT_DIR/lsun_bedroom.pt" https://www.dropbox.com/s/57bguxpr6by6l1x/lsun_bedroom.pt?dl=0

mkdir -p "$CHECKPOINT_DIR/ffhq_256_ncsnpp_continuous"
wget -O "$CHECKPOINT_DIR/ffhq_256_ncsnpp_continuous/checkpoint_48.pt" https://www.dropbox.com/s/9m86f0qxqop6pcu/checkpoint_48.pth?dl=0

mkdir -p "$CHECKPOINT_DIR/bedroom_ncsnpp_continuous"
wget -O "$CHECKPOINT_DIR/bedroom_ncsnpp_continuous/checkpoint_127.pt" https://www.dropbox.com/s/06osrjbqy4x8jlm/checkpoint_127.pth?dl=0

mkdir -p "$CHECKPOINT_DIR/AAPM_256_ncsnpp_continuous"
wget -O "$CHECKPOINT_DIR/AAPM_256_ncsnpp_continuous/checkpoint_185.pt" https://www.dropbox.com/s/prk5y3ltqcg6fmu/checkpoint_185.pth?dl=0

# 3. data
wget -O samples.zip https://www.dropbox.com/s/pvzww4wuilo4x62/samples.zip?dl=0
unzip samples.zip
rm samples.zip
