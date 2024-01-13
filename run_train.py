# -*- coding: utf-8 -*-
"""
author:     ZengZK
time  :     2024-01-11 18:09
"""
import os

if __name__ == '__main__':
    os.system('python train.py --dataset_name blender --root_dir /tianhe01/NeRF_Data/nerf_synthetic/lego --N_importance 64 --img_wh 400 400 --noise_std 0 --num_epochs 16 --batch_size 1024 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 --exp_name exp')
