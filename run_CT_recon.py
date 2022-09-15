import torch
# from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np
import controllable_generation

from utils import restore_checkpoint, show_samples_gray, clear, clear_color, \
    lambda_schedule_const, lambda_schedule_linear
from pathlib import Path
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import datasets
import time
# for radon
from physics.ct import CT
import matplotlib.pyplot as plt


###############################################
# Configurations
###############################################
solver = 'MCG'
config_name = 'AAPM_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 185
N = num_scales

root = './samples/CT'

# Parameters for the inverse problem
sparsity = 6
num_proj = 180 // sparsity  # 180 / 6 = 30
det_spacing = 1.0
size = 256
det_count = int((size * (2*torch.ones(1)).sqrt()).ceil()) # ceil(size * \sqrt{2})

schedule = 'linear'
start_lamb = 1.0
end_lamb = 0.6

num_posterior_sample = 1

if schedule == 'const':
    lamb_schedule = lambda_schedule_const(lamb=start_lamb)
elif schedule == 'linear':
    lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
else:
    NotImplementedError(f"Given schedule {schedule} not implemented yet!")

freq = 1

if sde.lower() == 'vesde':
    from configs.ve import AAPM_256_ncsnpp_continuous as configs
    ckpt_filename = f"./checkpoints/{config_name}/checkpoint_{ckpt_num}.pth"
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5

batch_size = 1
config.training.batch_size = batch_size
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1

batch_size = 1
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

# optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
# state = dict(step=0, optimizer=optimizer,
#              model=score_model, ema=ema)
state = dict(step=0, model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
ema.copy_to(score_model.parameters())

idx = 100
filename = Path(root) / (str(idx).zfill(4) + '.npy')
# Specify save directory for saving generated samples
save_root = Path(f'./results/SV-CT/m{180/sparsity}/{idx}/{solver}')
save_root.mkdir(parents=True, exist_ok=True)

irl_types = ['input', 'recon', 'label']
for t in irl_types:
    save_root_f = save_root / t
    save_root_f.mkdir(parents=True, exist_ok=True)

# Read data
img = torch.from_numpy(np.load(filename))
h, w = img.shape
img = img.view(1, 1, h, w)
img = img.to(config.device)

plt.imsave(save_root / 'label' / f'{str(idx).zfill(4)}.png', clear(img), cmap='gray')
plt.imsave(save_root / 'label' / f'{str(idx).zfill(4)}_clip.png', np.clip(clear(img), 0.1, 1.0), cmap='gray')

# full
angles = np.linspace(0, np.pi, 180, endpoint=False)
radon = CT(img_width=size, radon_view=num_proj, circle=False, device=config.device)
radon_all = CT(img_width=size, radon_view=180, circle=False, device=config.device)

mask = torch.zeros([batch_size, 1, det_count, 180]).to(config.device)
mask[..., ::sparsity] = 1

# Dimension Reducing (DR)
sinogram = radon.A(img)

# Dimension Preserving (DP)
sinogram_full = radon_all.A(img) * mask

# FBP
fbp = radon.A_dagger(sinogram)
plt.imsave(str(save_root / 'input' / f'FBP.png'), clear(fbp), cmap='gray')
if solver == 'MCG':
    pc_MCG = controllable_generation.get_pc_radon_MCG(sde,
                                                      predictor, corrector,
                                                      inverse_scaler,
                                                      snr=snr,
                                                      n_steps=n_steps,
                                                      probability_flow=probability_flow,
                                                      continuous=config.training.continuous,
                                                      denoise=True,
                                                      radon=radon,
                                                      radon_all=radon_all,
                                                      weight=0.1,
                                                      save_progress=False,
                                                      save_root=save_root,
                                                      lamb_schedule=lamb_schedule,
                                                      mask=mask)
    x = pc_MCG(score_model, scaler(img), measurement=sinogram)
elif solver == 'song':
    pc_song = controllable_generation.get_pc_radon_song(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        save_progress=True,
                                                        save_root=save_root,
                                                        denoise=True,
                                                        radon=radon_all,
                                                        lamb=0.7)
    x = pc_song(score_model, scaler(img), mask, measurement=sinogram_full)

# Recon
plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}.png'), clear(x), cmap='gray')
plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}_clip.png'), np.clip(clear(x), 0.1, 1.0), cmap='gray')