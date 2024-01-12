# -*- coding: utf-8 -*-
"""
author:     tianhe
time  :     2023-03-01 13:50
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default=r'F:\1_Code\1_Python\nerf_networks\results\blender-lego\tensorboard', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    os.system(f'tensorboard --logdir="{args.logdir}"')
