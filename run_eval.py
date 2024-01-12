# -*- coding: utf-8 -*-
"""
author:     ZengZK
time  :     2024-01-11 18:09
"""
import os

if __name__ == '__main__':
    os.system('python eval.py --dataset_name blender --root_dir F:/2_Datasets/NeRF/nerf_synthetic/lego --scene_name lego --img_wh 400 400 --N_importance 64 --ckpt_path F:/1_Code/1_Python/nerf_networks/results/lego-official/lego.ckpt')
