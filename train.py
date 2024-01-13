# -*- coding: utf-8 -*-
"""
author:     ZengZK
time  :     2024-01-11 16:46
"""
import argparse
import os
import random
import logging
from applog import log
from collections import defaultdict
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.losses import MSELoss
from models.nerf import Embedding, NeRF
from models.rendering import render_rays
from datasets import dataset_dict
from utils import get_optimizer, get_scheduler, get_learning_rate, visualize_depth
from utils.metrics import *


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')

    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse'],
                        help='loss to use')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()


def init_seed(seed=1008):
    """
    固定随机种子，复现模型结果
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 设置为True会选择最快的cudnn提供的卷积实现算法
    torch.backends.cudnn.deterministic = True  # 设置为True会选择默认的卷积算法


def decode_batch(batch):
    rays = batch['rays']  # (B, 8)
    rgbs = batch['rgbs']  # (B, 3)
    return rays, rgbs


def infer_by_models(models, embeddings, rays, hparams, white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, hparams.chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i + hparams.chunk],
                        hparams.N_samples,
                        hparams.use_disp,
                        hparams.perturb,
                        hparams.noise_std,
                        hparams.N_importance,
                        hparams.chunk,  # chunk size is effective in val mode
                        white_back)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def train_epoch(epoch, train_loader, models, embeddings, hparams, loss_func, optimizer, white_back, writer):
    for model in models:
        model.train()

    train_losses = []
    train_psnrs = []
    lr = get_learning_rate(optimizer)

    # train
    for batch_idx, data_in in enumerate(train_loader):
        rays, rgbs = decode_batch(data_in)
        rays, rgbs = rays.cuda(), rgbs.cuda()
        results = infer_by_models(models, embeddings, rays, hparams, white_back)
        train_loss = loss_func(results, rgbs)

        # 每次计算梯度前，将上一次梯度置零
        optimizer.zero_grad()
        # 计算梯度
        train_loss.backward()
        # 更新权重
        optimizer.step()

        # 指标计算
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        train_psnr = psnr(results[f'rgb_{typ}'], rgbs)

        # 保存指标
        train_losses.append(train_loss.item())
        train_psnrs.append(train_psnr.item())

        # 打印输出
        if (batch_idx + 1) % 100 == 0:
            logging.info(f'=> Epoch: [{epoch + 1:>3}/{hparams.num_epochs:>3}], Step [{batch_idx + 1:>5}/{len(train_loader):>5}], '
                         f'train loss: {train_loss:.5f}, train psnr: {train_psnr:.5f}, '
                         f'model.training: {models[0].training} {models[1].training}')

    mean_train_loss = np.mean(train_losses)
    mean_train_psnr = np.mean(train_psnrs)
    logging.info(f'=> Epoch: [{epoch + 1:>3}/{hparams.num_epochs:>3}], '
                 f'mean train loss: {mean_train_loss:.5f}, mean train psnr: {mean_train_psnr:.5f}, '
                 f'lr: {lr:.5f}, '
                 f'model.training: {models[0].training} {models[1].training}')
    writer.add_scalar(f'learning rate', lr, epoch + 1)
    writer.add_scalars(f'loss', {'train loss': mean_train_loss}, epoch + 1)
    writer.add_scalars(f'psnr', {'train psnr': mean_train_psnr}, epoch + 1)


@torch.no_grad()
def val_epoch(epoch, val_loader, models, embeddings, hparams, loss_func, white_back, writer):
    for model in models:
        model.eval()

    val_losses = []
    val_psnrs = []

    # val
    for batch_idx, data_in in enumerate(val_loader):
        rays, rgbs = decode_batch(data_in)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        rays, rgbs = rays.cuda(), rgbs.cuda()
        results = infer_by_models(models, embeddings, rays, hparams, white_back)
        val_loss = loss_func(results, rgbs)

        # 指标计算
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        val_psnr = psnr(results[f'rgb_{typ}'], rgbs)

        # 保存指标
        val_losses.append(val_loss.item())
        val_psnrs.append(val_psnr.item())

        # 打印输出
        if batch_idx == 0 or batch_idx == 1:
            W, H = hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torchvision.utils.make_grid([img_gt, img, depth], nrow=3, padding=10)
            writer.add_image(f'GT_pred_depth/{batch_idx}', stack, epoch + 1)

        if (batch_idx + 1) % 1 == 0:
            logging.info(f'=> Epoch: [{epoch + 1:>4}], Step [{batch_idx + 1:>5}/{len(val_loader):>5}], '
                         f'val loss: {val_loss:.5f}, val psnr: {val_psnr:.5f}, '
                         f'model.training: {models[0].training} {models[1].training}')

    mean_val_loss = np.mean(val_losses)
    mean_val_psnr = np.mean(val_psnrs)
    logging.info(f'=> Epoch: [{epoch + 1:>3}/{hparams.num_epochs:>3}], '
                 f'mean val loss: {mean_val_loss:.5f}, mean val psnr: {mean_val_psnr:.5f}, '
                 f'model.training: {models[0].training} {models[1].training}')
    writer.add_scalars(f'loss', {'val loss': mean_val_loss}, epoch + 1)
    writer.add_scalars(f'psnr', {'val psnr': mean_val_psnr}, epoch + 1)
    return mean_val_loss, mean_val_psnr


def main():
    # param
    print('=> =========================== Get Params ===========================')
    hparams = get_opts()

    print('=> =========================== Create Dir ===========================')
    log_dir = os.path.join('results', f'{hparams.dataset_name}-{os.path.split(hparams.root_dir)[-1]}', 'log')
    tensorboard_dir = os.path.join('results', f'{hparams.dataset_name}-{os.path.split(hparams.root_dir)[-1]}', 'tensorboard')
    checkpoint_dir = os.path.join('results', f'{hparams.dataset_name}-{os.path.split(hparams.root_dir)[-1]}', 'checkpoint')
    if os.path.exists(log_dir):
        print(f'-- {log_dir} exist')
        exit(0)
    if os.path.exists(tensorboard_dir):
        print(f'-- {tensorboard_dir} exist')
        exit(0)
    if os.path.exists(checkpoint_dir):
        print(f'-- {checkpoint_dir} exist')
        exit(0)
    print(f'-- create log dir : {log_dir}')
    print(f'-- create tensorboard dir : {tensorboard_dir}')
    print(f'-- create checkpoint dir : {checkpoint_dir}')
    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(tensorboard_dir, exist_ok=False)
    os.makedirs(checkpoint_dir, exist_ok=False)
    # Init log
    log.init(log_dir)
    logging.info('=> ======================== Start Training ==========================')
    # Print Args
    logging.info(f'=> {hparams}')
    logging.info('=> Launch tensorboard')
    writer = SummaryWriter(tensorboard_dir)

    logging.info('=> =========================== Init Seed ============================')
    # For reproducibility
    init_seed()

    # model
    logging.info('=> =========================== Load Model ===========================')
    embedding_xyz = Embedding(3, 10)  # 10 is the default number
    embedding_dir = Embedding(3, 4)  # 4 is the default number
    embeddings = [embedding_xyz, embedding_dir]
    nerf_coarse = NeRF().cuda()
    models = [nerf_coarse]
    if hparams.N_importance > 0:
        nerf_fine = NeRF().cuda()
        models += [nerf_fine]

    logging.info(f'=> models len: {len(models)}')

    # data
    logging.info('=> =========================== Load Data ============================')
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
              'img_wh': tuple(hparams.img_wh)}
    if hparams.dataset_name == 'llff':
        kwargs['spheric_poses'] = hparams.spheric_poses
        kwargs['val_num'] = hparams.num_gpus
    train_dataset = dataset(split='train', **kwargs)
    val_dataset = dataset(split='val', **kwargs)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers=4,
                              batch_size=hparams.batch_size,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            num_workers=4,
                            batch_size=1,  # validate one image (H*W rays) at a time
                            pin_memory=True)
    logging.info(f'=> train dataloader len: {len(train_loader.dataset)}')
    logging.info(f'=> val dataloader len: {len(val_loader.dataset)}')

    # optim
    logging.info('=> ========================= Init Optimizer =========================')
    optimizer = get_optimizer(hparams, models)
    scheduler = get_scheduler(hparams, optimizer)

    # loss func
    logging.info('=> ========================= Init Loss Func =========================')
    loss_func = MSELoss()

    for epoch in range(hparams.num_epochs):
        logging.info('=> ============================ Training ============================')
        train_epoch(epoch=epoch,
                    train_loader=train_loader,
                    models=models,
                    embeddings=embeddings,
                    hparams=hparams,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    white_back=train_dataset.white_back,
                    writer=writer)

        logging.info('=> =========================== Validating ===========================')
        val_loss, val_psnr = \
            val_epoch(epoch=epoch,
                      val_loader=val_loader, models=models,
                      embeddings=embeddings,
                      hparams=hparams,
                      loss_func=loss_func,
                      white_back=train_dataset.white_back,
                      writer=writer)

        logging.info('=> =========================== Update LR ============================')
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        logging.info('=> ======================== Save Checkpoint =========================')
        checkpoint_filename = checkpoint_dir + f'/checkpoint_{epoch + 1:>03}_loss{val_loss:.5f}_psnr{val_psnr:.5f}_coarse.ckpt'
        torch.save({
            'epoch': epoch + 1,
            'state_dict': models[0].state_dict(),
        }, checkpoint_filename)
        if hparams.N_importance > 0:
            checkpoint_filename = checkpoint_dir + f'/checkpoint_{epoch + 1:>03}_loss{val_loss:.5f}_psnr{val_psnr:.5f}_fine.ckpt'
            torch.save({
                'epoch': epoch + 1,
                'state_dict': models[1].state_dict(),
            }, checkpoint_filename)


if __name__ == '__main__':
    main()
