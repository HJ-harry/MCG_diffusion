import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
from pathlib import Path
from utils import clear_color, normalize_np, clear, prepare_im
import math


# added
def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # img
    fname = args.image_num
    load_dir = Path(args.base_samples) / f'{fname}.png'
    ref_img = prepare_im(load_dir, args.image_size, next(model.parameters()).device)

    # mask
    mask = plt.imread(Path(args.mask_dir) / args.mask_type / f'{fname}.png')
    mask = th.from_numpy(mask[..., :3]).permute(2, 0, 1).unsqueeze(dim=0).to(ref_img.device)

    down_ref_img = ref_img * mask
    model_kwargs = {'ref_img': down_ref_img}

    out_path = Path(args.save_dir) / f'{args.mask_type}' / fname
    out_path.mkdir(parents=True, exist_ok=True)

    plt.imsave(out_path / f'input.png', clear_color(down_ref_img))
    plt.imsave(out_path / f'label.png', clear_color(ref_img))
    for j in range(args.num_samples):
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            mask=mask,
            save_root=out_path,
            sample_method=args.sample_method,
            progress=True,
        )
        plt.imsave(out_path / f'diffusion_{j}.png', clear_color(sample))

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4,
        batch_size=1,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_method', type=str, default='MCG', help='One of [vanilla, MCG, repaint]')
    parser.add_argument('--mask_type', type=str, default='box', help='One of [box, random]')
    parser.add_argument('--mask_dir', type=str, default='./samples/mask',
                        help='directory where the pre-sampled masks are located')
    parser.add_argument('--repeat_steps', type=int, default=20, help='For REPAINT, number of repeat steps')
    parser.add_argument('--image_num', type=str, default='00000', help='Image to be used for sampling')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory where the results will be saved')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()