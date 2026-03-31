import os
import sys

sys.path.append('../')

import gc
import logging
import random
import sys
import time

import numpy as np
import torch
import wandb
from utils import worker_init_fn, register_signal_handler
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import DiceMetric, PSNRMetric, SSIMMetric
from models import Warp, Composite, VecIntegrate, build_loss, build_metrics, build_registration_head, \
    build_flow_estimator

from run_iter import run_iter
from utils import load_data_01 as load_data

torch.backends.cudnn.deterministic = True


def train(train_cfg_file: str, random_seed: int = 42):
    ##### configuration and setup #####
    cfg = Config.fromfile(train_cfg_file)
    cfg.update(dict(random_seed=random_seed))
    wandb.init(project=cfg.project, name=cfg.name, config=dict(cfg))
    model_dir = os.path.join(cfg.out_path, cfg.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.out_path, 'train_configs.py'))
    cfg.update(dict(amp_dtype=getattr(torch, cfg.amp_dtype)))

    ##### build model and utils like registration head/warp/composition #####
    model = build_flow_estimator(cfg.model_cfg).to(cfg.device)
    if cfg.load_model:
        model.load_state_dict(
            torch.load(cfg.load_model, map_location=torch.device(cfg.device)))
        print(f'model from {cfg.load_model} loaded.')
    wandb.watch(model, log='gradients', log_freq=100, log_graph=True)

    warp = Warp(image_size=cfg.registration_cfg.image_size,
                interp_mode=cfg.registration_cfg.interp_mode).to(cfg.device)
    compose = Composite(image_size=cfg.registration_cfg.image_size,
                        interp_mode=cfg.registration_cfg.interp_mode).to(cfg.device)
    reg_head = build_registration_head(cfg.registration_cfg).to(cfg.device)
    vecint = VecIntegrate(image_size=cfg.vecint_cfg.image_size,
                          num_steps=cfg.vecint_cfg.num_steps,
                          interp_mode=cfg.vecint_cfg.interp_mode).to(cfg.device) if cfg.vecint_cfg else None


    ##### dataset loading
    train_dataset = load_data(cfg,
                            cache_rate=cfg.cache_rate,
                            num_workers=cfg.num_workers)
    train_loader = DataLoader(train_dataset,
                              batch_size=1, # real batch size depends of the number of synthetic triples (sampled t)
                              shuffle=True,
                              drop_last=False,
                              worker_init_fn=worker_init_fn)
    
    print(f'total:{len(train_loader)}')

    ##### build loss functions and metrics #####
    loss_funcs = dict(
        sim=build_loss(cfg.sim_loss_cfg).to(cfg.device),
        reg=build_loss(cfg.reg_loss_cfg).to(cfg.device),
        flow=build_loss(cfg.flow_loss_cfg).to(cfg.device),
        gradicon=build_loss(cfg.gradicon_loss_cfg).to(cfg.device)
    )
    metric_funcs = dict(dice=DiceMetric(include_background=False, reduction='mean'),
                        ssim=SSIMMetric(data_range=torch.tensor(1.0),
                              spatial_dims=3)._compute_metric,
                        psnr=PSNRMetric(max_val=1.0)._compute_metric,
                        jacdet=build_metrics(dict(type='fg_sdlogjac')))

    ##### optimizer, lr_scheduler, scaler for amp #####
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=0,
                                 amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                          gamma=cfg.lr_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    epoch = cfg.start_epoch
    register_signal_handler(lambda: epoch, model_dir, model, optimizer, lr_scheduler, scaler)

    # epoch loop
    for epoch in range(cfg.start_epoch, cfg.max_epochs):
        print('Epoch:', epoch)
        phase = 'train'
        model.train()
        run_iter(cfg,
                 model,
                 train_loader,
                 loss_funcs,
                 metric_funcs,
                 vecint,
                 warp,
                 compose,
                 scaler,
                 optimizer,
                 len(train_loader) * epoch,
                 phase)

        lr_scheduler.step()

        if epoch % cfg.save_interval == 0 and epoch != cfg.start_epoch:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, '%04d.pth' % epoch))

    torch.save(model.state_dict(),
               os.path.join(model_dir, '%04d.pth' % cfg.max_epochs))

    max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print("[+] Maximum memory:\t{:.2f}GB".format(max_mem_mb))
    max_mem_re = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print("[+] Maximum memory:\t{:.2f}GB".format(max_mem_re))


if __name__ == '__main__':
    import pathlib

    import configargparse

    from utils import set_seed

    p = configargparse.ArgParser()
    p.add_argument('--train-config',
                   required=True,
                   type=lambda f: pathlib.Path(f).absolute(),
                   help='path of train configure file')
    p.add_argument('--random-seed',
                   '-seed',
                   required=True,
                   type=int,
                   help='random seed')
    args = p.parse_args()
    set_seed(args.random_seed)
    train(args.train_config, args.random_seed)