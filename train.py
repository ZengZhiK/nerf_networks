# -*- coding: utf-8 -*-
"""
author:     ZengZK
time  :     2024-01-11 16:46
"""
import argparse
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from models.losses import MSELoss
from models.nerf import Embedding, NeRF
from models.rendering import render_rays
from datasets import dataset_dict
from utils import get_optimizer, get_scheduler, get_learning_rate, visualize_depth
# metrics
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


def train_epoch(epoch, train_loader, models, embeddings, hparams, loss_func, optimizer, scheduler, white_back):
    for model in models:
        model.train()

    # train
    for batch_idx, data_in in enumerate(train_loader):
        rays, rgbs = decode_batch(data_in)
        rays, rgbs = rays.cuda(), rgbs.cuda()
        results = infer_by_models(models, embeddings, rays, hparams, white_back)
        train_loss = loss_func(results, rgbs)

        # 保存学习率
        lr = get_learning_rate(optimizer)
        # 每次计算梯度前，将上一次梯度置零
        optimizer.zero_grad()
        # 计算梯度
        train_loss.backward()
        # 更新权重
        optimizer.step()
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 指标计算
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        with torch.no_grad():
            train_psnr = psnr(results[f'rgb_{typ}'], rgbs)

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch: [{epoch + 1:>4}], Step [{batch_idx + 1:>5}/{len(train_loader):>5}], '
                  f'train loss: {train_loss:.4f}, train psnr: {train_psnr:.4f}, '
                  f'model train: {models[0].training} {models[1].training}')


def val_epoch(epoch, val_loader, models, embeddings, hparams, loss_func, white_back):
    for model in models:
        model.eval()

    # val
    for batch_idx, data_in in enumerate(val_loader):
        rays, rgbs = decode_batch(data_in)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        rays, rgbs = rays.cuda(), rgbs.cuda()
        with torch.no_grad():
            results = infer_by_models(models, embeddings, rays, hparams, white_back)
        val_loss = loss_func(results, rgbs)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        with torch.no_grad():
            val_psnr = psnr(results[f'rgb_{typ}'], rgbs)

        if batch_idx == 0:
            W, H = hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            # self.logger.experiment.add_images('val/GT_pred_depth', stack, self.global_step)

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch: [{epoch + 1:>4}], Step [{batch_idx + 1:>5}/{len(val_loader):>5}], '
                  f'val loss: {val_loss:.4f}, val psnr: {val_psnr:.4f}, '
                  f'model train: {models[0].training} {models[1].training}')


def main():
    # param
    print('=> =========================== Get Params ===========================')
    hparams = get_opts()

    # model
    print('=> =========================== Load Model ===========================')
    embedding_xyz = Embedding(3, 10)  # 10 is the default number
    embedding_dir = Embedding(3, 4)  # 4 is the default number
    embeddings = [embedding_xyz, embedding_dir]
    nerf_coarse = NeRF().cuda()
    models = [nerf_coarse]
    if hparams.N_importance > 0:
        nerf_fine = NeRF().cuda()
        models += [nerf_fine]

    # data
    print('=> =========================== Load Data ============================')
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

    # optim
    print('=> ========================= Init Optimizer =========================')
    optimizer = get_optimizer(hparams, models)
    scheduler = get_scheduler(hparams, optimizer)

    # loss func
    print('=> ========================= Init Loss Func =========================')
    loss_func = MSELoss()

    print('=> ============================ Training ============================')
    for epoch in range(hparams.num_epochs):
        train_epoch(epoch=epoch,
                    train_loader=train_loader,
                    models=models,
                    embeddings=embeddings,
                    hparams=hparams,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    white_back=train_dataset.white_back)

        val_epoch(epoch=epoch,
                  val_loader=val_loader, models=models,
                  embeddings=embeddings,
                  hparams=hparams,
                  loss_func=loss_func,
                  white_back=train_dataset.white_back)


if __name__ == '__main__':
    main()
