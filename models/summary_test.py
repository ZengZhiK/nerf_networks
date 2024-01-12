# -*- coding: utf-8 -*-
"""
author:     tianhe
time  :     2023-02-10 15:39
"""
import numpy as np
import torch
from torchsummary import summary
from models.nerf import NeRF

if __name__ == '__main__':
    model = NeRF()
    summary(model, [(1, 256, 512)])
