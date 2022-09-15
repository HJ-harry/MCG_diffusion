# Manifold Constrained Gradient Diffusion

Official PyTorch implementation of the NeurIPS 2022 paper "[Improving Diffusion Models for Inverse Problems using Manifold Constraints](https://arxiv.org/abs/2206.00941)"

For each task, we additionally provide some re-implementations of diffusion model-based inverse problem solvers.


## Getting Started

### Setting the environment

Our code was tested on the following environment
- Ubuntu 20.04
- CUDA 10.2
- PyTorch 1.6.0

We provide the install script in ```install.sh```. Be sure to have conda ready,
as we will be using conda to build the environment. Once ready, simply run the following command:
```bash
source install.sh
```
The above command will create a conda environment, and install the dependencies listed in ```requirements.txt```.

### Pretrained checkpoints

The install script will automatically download the pre-trained checkpoints and place it in the
```checkpoints``` directory. Alternatively, you may download the pre-trained 
checkpoints used for each task from the links below.

|              | FFHQ 256x256                                                             | ImageNet 256x256                                                      | LSUN-bedroom                                                              | AAPM                                                                      | Original repository                                            |
|--------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------|
| Inpainting   | [Link](https://www.dropbox.com/s/4r8r6o2n1pumzmg/ffhq_10m.pt?dl=1)       | [Link](https://www.dropbox.com/s/rtit2qsb353262t/imagenet256.pt?dl=0) | [Link](https://www.dropbox.com/s/57bguxpr6by6l1x/lsun_bedroom.pt?dl=1)    | -                                                                         | [Guided diffusion](https://github.com/openai/guided-diffusion) |
| Colorization | [Link](https://www.dropbox.com/s/9m86f0qxqop6pcu/checkpoint_48.pth?dl=1) | -                                                                     | [Link](https://www.dropbox.com/s/06osrjbqy4x8jlm/checkpoint_127.pth?dl=1) | -                                                                         | [Score-SDE](https://github.com/yang-song/score_sde_pytorch)    |
| SV-CT        | -                                                                        | -                                                                     | -                                                                         | [Link](https://www.dropbox.com/s/prk5y3ltqcg6fmu/checkpoint_185.pth?dl=1) | -                                                              |


## Solving Inverse Problems with MCG

For all tasks, we provide some sample data, which is contained in the `samples` folder,
which will be automatically downloaded with the install script. You may alternatively get the data
[here](https://www.dropbox.com/s/pvzww4wuilo4x62/samples.zip?dl=1). All results will be saved in the `results` folder.


### Inpainting

Run the following command to perform inpainting with the default configurations.
```bash
bash run_inpainting.sh
```
You may try other sampling strategies other than `MCG` by changing the flag ```--sample_method```
to `vanilla` ([score-SDE](https://github.com/yang-song/score_sde_pytorch)), or `repaint` ([RePAINT](https://github.com/andreas128/RePaint)).

### Colorization

Run the following command. Change the parameters directly in the python script as needed.
```
python run_colorization.py
```

### Sparse-view CT reconstruction (SV-CT)

Run the following command. Change the parameters directly in the python script as needed.
```
python run_CT_recon.py
```
We additionally provide our re-implementation of this [paper](https://openreview.net/forum?id=vaRCHVj0uGI). Set `solver='song'`
when you wish to run the solver introduced in [Song et al.](https://openreview.net/forum?id=vaRCHVj0uGI).

