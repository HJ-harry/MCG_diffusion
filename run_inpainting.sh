#!/bin/bash

MODEL_FLAGS="--attention_resolutions 16 \
--class_cond False \
--diffusion_steps 1000 \
--image_size 256 \
--learn_sigma True \
--noise_schedule linear \
--num_channels 128 \
--num_head_channels 64 \
--num_res_blocks 1 \
--resblock_updown True \
--use_fp16 False \
--use_scale_shift_norm True"

python scripts/sample_inpainting.py \
 $MODEL_FLAGS \
 --timestep_respacing 1000 \
 --use_ddim False \
 --num_samples 4 \
 --model_path checkpoints/ffhq_10m.pt \
 --sample_method MCG \
 --repeat_steps 20 \
 --base_samples ./samples/image/FFHQ \
 --mask_dir ./samples/mask \
 --mask_type box \
 --save_dir ./results/inpainting/ffhq