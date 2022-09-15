import torch
import tensorflow as tf
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sporco.metric import gmsd, mse
from scipy.ndimage import gaussian_laplace
import functools


def clear_color(x):
  x = x.detach().cpu().squeeze().numpy()
  return normalize_np(np.transpose(x, (1, 2, 0)))

def clear(x):
  x = x.detach().cpu().squeeze().numpy()
  return x


def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False, skip_optimizer=False):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.error(f"No checkpoint found at {ckpt_dir}. "
                  f"Returned the same state as input")
    FileNotFoundError(f'No such checkpoint: {ckpt_dir} found!')
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    if not skip_optimizer:
      state['optimizer'].load_state_dict(loaded_state['optimizer'])
    loaded_model_state = loaded_state['model']
    if skip_sigma:
      loaded_model_state.pop('module.sigmas')

    state['model'].load_state_dict(loaded_model_state, strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    print(f'loaded checkpoint dir from {ckpt_dir}')
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


"""
Helper functions for new types of inverse problems
"""


def crop_center(img, cropx, cropy):
  c, y, x = img.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)
  return img[:, starty:starty + cropy, startx:startx + cropx]


def normalize(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= torch.min(img)
  img /= torch.max(img)
  return img

def normalize_np(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= np.min(img)
  img /= np.max(img)
  return img


def normalize_complex(img):
  """ normalizes the magnitude of complex-valued image to range [0, 1] """
  abs_img = normalize(torch.abs(img))
  # ang_img = torch.angle(img)
  ang_img = normalize(torch.angle(img))
  return abs_img * torch.exp(1j * ang_img)


class lambda_schedule:
  def __init__(self, total=2000):
    self.total = total

  def get_current_lambda(self, i):
    pass


class lambda_schedule_linear(lambda_schedule):
  def __init__(self, start_lamb=1.0, end_lamb=0.0):
    super().__init__()
    self.start_lamb = start_lamb
    self.end_lamb = end_lamb

  def get_current_lambda(self, i):
    return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
  def __init__(self, lamb=1.0):
    super().__init__()
    self.lamb = lamb

  def get_current_lambda(self, i):
    return self.lamb




def image_grid(x, sz=32):
  size = sz
  channels = 3
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img


def show_samples(x, sz=32):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x, sz)
  plt.figure(figsize=(8, 8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()


def image_grid_gray(x, size=32):
  img = x.reshape(-1, size, size)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size)).transpose((0, 2, 1, 3)).reshape((w * size, w * size))
  return img


def show_samples_gray(x, size=32, save=False, save_fname=None):
  x = x.detach().cpu().numpy()
  img = image_grid_gray(x, size=size)
  plt.figure(figsize=(8, 8))
  plt.axis('off')
  plt.imshow(img, cmap='gray')
  plt.show()
  if save:
    plt.imsave(save_fname, img, cmap='gray')


def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
  mux_in = size ** 2
  if type.endswith('2d'):
    Nsamp = mux_in // acc_factor
  elif type.endswith('1d'):
    Nsamp = size // acc_factor
  if type == 'gaussian2d':
    mask = torch.zeros_like(img)
    cov_factor = size * (1.5 / 128)
    mean = [size // 2, size // 2]
    cov = [[size * cov_factor, 0], [0, size * cov_factor]]
    if fix:
      samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
    else:
      for i in range(batch_size):
        # sample different masks for batch
        samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
  elif type == 'uniformrandom2d':
    mask = torch.zeros_like(img)
    if fix:
      mask_vec = torch.zeros([1, size * size])
      samples = np.random.choice(size * size, int(Nsamp))
      mask_vec[:, samples] = 1
      mask_b = mask_vec.view(size, size)
      mask[:, ...] = mask_b
    else:
      for i in range(batch_size):
        # sample different masks for batch
        mask_vec = torch.zeros([1, size * size])
        samples = np.random.choice(size * size, int(Nsamp))
        mask_vec[:, samples] = 1
        mask_b = mask_vec.view(size, size)
        mask[i, ...] = mask_b
  elif type == 'gaussian1d':
    mask = torch.zeros_like(img)
    mean = size // 2
    std = size * (15.0 / 128)
    Nsamp_center = int(size * center_fraction)
    if fix:
      samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[... , int_samples] = 1
      c_from = size // 2 - Nsamp_center // 2
      mask[... , c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, :, int_samples] = 1
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from + Nsamp_center] = 1
  elif type == 'uniform1d':
    mask = torch.zeros_like(img)
    if fix:
      Nsamp_center = int(size * center_fraction)
      samples = np.random.choice(size, int(Nsamp - Nsamp_center))
      mask[..., samples] = 1
      # ACS region
      c_from = size // 2 - Nsamp_center // 2
      mask[..., c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        Nsamp_center = int(size * center_fraction)
        samples = np.random.choice(size, int(Nsamp - Nsamp_center))
        mask[i, :, :, samples] = 1
        # ACS region
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from+Nsamp_center] = 1
  else:
    NotImplementedError(f'Mask type {type} is currently not supported.')

  return mask


def kspace_to_nchw(tensor):
    """
    Convert torch tensor in (Slice, Coil, Height, Width, Complex) 5D format to
    (N, C, H, W) 4D format for processing by 2D CNNs.

    Complex indicates (real, imag) as 2 channels, the complex data format for Pytorch.

    C is the coils interleaved with real and imaginary values as separate channels.
    C is therefore always 2 * Coil.

    Singlecoil data is assumed to be in the 5D format with Coil = 1

    Args:
        tensor (torch.Tensor): Input data in 5D kspace tensor format.
    Returns:
        tensor (torch.Tensor): tensor in 4D NCHW format to be fed into a CNN.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 5
    s = tensor.shape
    assert s[-1] == 2
    tensor = tensor.permute(dims=(0, 1, 4, 2, 3)).reshape(shape=(s[0], 2 * s[1], s[2], s[3]))
    return tensor


def nchw_to_kspace(tensor):
  """
  Convert a torch tensor in (N, C, H, W) format to the (Slice, Coil, Height, Width, Complex) format.

  This function assumes that the real and imaginary values of a coil are always adjacent to one another in C.
  If the coil dimension is not divisible by 2, the function assumes that the input data is 'real' data,
  and thus pads the imaginary dimension as 0.
  """
  assert isinstance(tensor, torch.Tensor)
  assert tensor.dim() == 4
  s = tensor.shape
  if tensor.shape[1] == 1:
    imag_tensor = torch.zeros(s, device=tensor.device)
    tensor = torch.cat((tensor, imag_tensor), dim=1)
    s = tensor.shape
  tensor = tensor.view(size=(s[0], s[1] // 2, 2, s[2], s[3])).permute(dims=(0, 1, 3, 4, 2))
  return tensor


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def save_data(fname, arr):
  """ Save data as .npy and .png """
  np.save(fname + '.npy', arr)
  plt.imsave(fname + '.png', arr, cmap='gray')

def mean_std(vals: list):
  return mean(vals), stdev(vals)

def cal_metric(comp, label):
  LoG = functools.partial(gaussian_laplace, sigma=1.5)
  psnr_val = peak_signal_noise_ratio(comp, label)
  ssim_val = structural_similarity(comp, label)
  hfen_val = mse(LoG(comp), LoG(label))
  gmsd_val = gmsd(label, comp)
  return psnr_val, ssim_val, hfen_val, gmsd_val

def prepare_im(load_dir, image_size, device):
  ref_img = torch.from_numpy(plt.imread(load_dir)[:, :, :3]).to(device)
  ref_img = ref_img.permute(2, 0, 1)
  ref_img = ref_img.view(1, 3, image_size, image_size)
  ref_img = ref_img * 2 - 1
  return ref_img