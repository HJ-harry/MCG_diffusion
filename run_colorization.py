import matplotlib.pyplot as plt
import matplotlib
import torch
from models.ema import ExponentialMovingAverage

from pathlib import Path
import controllable_generation
from utils import restore_checkpoint, clear_color, clear

import models
from models import utils as mutils
from models import ncsnpp
import sampling
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import datasets

problem = 'colorization'
num_scales = 2000
config_name = 'bedroom_ncsnpp_continuous'
sde = 'VESDE'
if sde.lower() == 'vesde':
    from configs.ve import bedroom_ncsnpp_continuous as configs

    ckpt_filename = f"./checkpoints/{config_name}/checkpoint_127.pth"
    config = configs.get_config()
    config.model.num_scales = num_scales
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5

batch_size = 1
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_optimizer=True)
ema.copy_to(score_model.parameters())

predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
snr = 0.16
n_steps = 1
probability_flow = False

load_root = Path('./samples/image/LSUN-bedroom')
save_root = Path(f'./results/colorization')
save_root.mkdir(parents=True, exist_ok=True)

pc_colorizer_grad = controllable_generation.get_pc_colorizer_grad(sde,
                                                                  predictor, corrector,
                                                                  inverse_scaler,
                                                                  snr=snr,
                                                                  n_steps=n_steps,
                                                                  probability_flow=probability_flow,
                                                                  continuous=config.training.continuous,
                                                                  weight=0.1,
                                                                  denoise=True)
idx = 0
fname = str(idx).zfill(5)
img = plt.imread(load_root / f'{fname}.png')[:, :, :3]
img = torch.from_numpy(img)
img = img.view(1, 256, 256, 3)
img = img.permute(0, 3, 1, 2).to(config.device)
plt.imsave(save_root / f'label.png', clear_color(img))

gray_scale_img = torch.mean(img, dim=1, keepdims=True).repeat(1, 3, 1, 1)
plt.imsave(save_root / 'input.png', clear_color(gray_scale_img))

x = pc_colorizer_grad(score_model, scaler(gray_scale_img))
plt.imsave(save_root / 'recon.png', clear_color(x))
