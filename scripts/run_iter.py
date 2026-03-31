import sys
import time

sys.path.append('../')
from itertools import combinations, permutations
from random import choice, sample, shuffle
from typing import (Callable, Dict, List, Literal, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import torch
import torch.nn as nn
import wandb
from mmengine import Config, ConfigDict
from monai.data import DataLoader
from torch.optim import Optimizer

from utils import optional_context

CFG = Union[dict, Config, ConfigDict]


def run_iter(cfg: CFG,
             model: nn.Module,
             dataloader: DataLoader,
             loss_funcs: Sequence[Callable],
             metric_funcs: Sequence[Callable],
             vecint: nn.Module,
             warp: nn.Module,
             compose: nn.Module,
             scaler: torch.cuda.amp.GradScaler,
             optimizer: Optimizer,
             epoch_iter: int,
             phase: str):
    logging_dict = dict()
    dataiter = iter(dataloader)
    k = 0
    for _ in range(len(dataloader)):
        try:
            data = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            data = next(dataiter)

        logging_dict.update({'iter': epoch_iter + k})
        k += 1

            # sample t in [0.1, 0.9] or 1/t in [1.1, 10.0]
            if cfg.batch_size == 1:
                t = torch.rand((cfg.batch_size,), device=cfg.device, dtype=torch.float32)
                if np.random.rand() < 0.5:
                    t = 0.1 + 0.8 * t
                else:
                    t = 1.1 + 8.9 * t
                    t = 1.0 / t
            elif cfg.batch_size >= 2 and cfg.batch_size % 2 == 0:
                t_0 = torch.rand((cfg.batch_size // 2,), device=cfg.device, dtype=torch.float32)
                t_1 = torch.rand((cfg.batch_size // 2,), device=cfg.device, dtype=torch.float32)
                t_0 = 0.1 + 0.8 * t_0
                t_1 = 1.1 + 8.9 * t_1
                t_1 = 1.0 / t_1
                t = torch.cat([t_0, t_1], dim=0)
            else:
                raise ValueError(f'batch size should be 1 or even number, got {cfg.batch_size}')
            print('sampled t:', t)

            # randomly set baseline/endpoint scans as source or target
            source, target = [data[f't{i}'].float().to(cfg.device) for i in np.random.permutation(2)]
            # if oversample, do pairwise registration only
            oversample = (np.random.rand() < cfg.oversample_rate) or torch.any(t == 1.0)

            total_loss = 0.0
            with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=cfg.use_amp):
                # t=1 -> pairwise registration
                one = torch.ones((1,), device=cfg.device, dtype=source.dtype)
                if oversample:
                ########################## pairwise registration #######################################
                    # from source to target
                    fwd_flow_st = model(source, target, one)
                    bck_flow_st = model(source, target, -one)
                    # from target to source
                    fwd_flow_ts = model(target, source, one)
                    bck_flow_ts = model(target, source, -one)

                    if vecint:
                        fwd_flow_st = vecint(fwd_flow_st)
                        bck_flow_st = vecint(bck_flow_st)
                        fwd_flow_ts = vecint(fwd_flow_ts)
                        bck_flow_ts = vecint(bck_flow_ts)

                    # flows warping source to target
                    src_tar_flows = torch.cat([fwd_flow_st, bck_flow_ts], dim=0)
                    # flows warping target to source
                    tar_src_flows = torch.cat([fwd_flow_ts, bck_flow_st], dim=0)

                    flows = torch.cat([src_tar_flows, tar_src_flows], dim=0)

                    source = source.expand(2, *[-1] * 4)
                    target = target.expand(2, *[-1] * 4)

                    # warp images
                    y_source = warp(source, src_tar_flows)
                    y_target = warp(target, tar_src_flows)

                    # losses: 2 pairs of bidirectional similarity + 4 pairs of inverse consistency (GradICON)
                    with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=False):
                        sim = 0.5 * (loss_funcs['sim'](y_source, target) + loss_funcs['sim'](y_target, source))
                    #
                    total_loss += cfg.sim_loss_cfg.weight * sim
                    #
                    logging_dict.update({'sim': sim.detach().cpu()})

                    with optional_context(cfg.gradicon_weight == 0.0, torch.no_grad()):
                        # GradICON
                        grad_icon = loss_funcs['gradicon'](
                            torch.cat([fwd_flow_st, fwd_flow_ts, fwd_flow_st, bck_flow_st], dim=0),
                            torch.cat([bck_flow_st, bck_flow_ts, fwd_flow_ts, bck_flow_ts], dim=0)
                        )
                        #
                        total_loss += cfg.gradicon_weight * grad_icon
                        #
                        logging_dict.update({'grad_icon': grad_icon.detach().cpu()})
                ########################################################################################

                else:
                    # otherwise, with inter-/extra-polation consistency constraints
                    fg = torch.where(source > 0, 1.0, 0.0).float().to(cfg.device)
                    # flow warping source to target
                    flow_one = model(source, target, one)
                    # flow warping target to source
                    flow_neg_one = model(source, target, -one)
                    # flow warping source(t=0) to t
                    flow_t = model(source, target, t)
                    # flow warping target(t=1) to t
                    flow_t_1 = model(source, target, t - 1.0)

                    if vecint:
                        flow_one = vecint(flow_one)
                        flow_neg_one = vecint(flow_neg_one)
                        flow_t = vecint(flow_t)
                        flow_t_1 = vecint(flow_t_1)

                    flows = torch.cat([flow_one, flow_neg_one], dim=0)

                    y_source = warp(source, flow_one)
                    y_target = warp(target, flow_neg_one)

                    source_t = warp(source.expand(cfg.batch_size, *[-1] * 4), flow_t)
                    target_t_1 = warp(target.expand(cfg.batch_size, *[-1] * 4), flow_t_1)

                    ########################### interpolation consistency constraint #######################
                    # image similarity
                    with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=False):
                        # registration pairs
                        #   y_source vs target
                        #   y_target vs source
                        #   source_t vs target_t_1 (interpolation)
                        sim = loss_funcs['sim'](
                            torch.cat([y_source, y_target, source_t], dim=0),
                            torch.cat([target, source, target_t_1], dim=0)
                        )
                    #
                    total_loss += cfg.sim_loss_cfg.weight * sim
                    #
                    logging_dict.update({'sim': sim.detach().cpu()})

                    # inverse consistency
                    with optional_context(cfg.gradicon_weight == 0.0, torch.no_grad()):
                        # GradICON
                        grad_icon = loss_funcs['gradicon'](flow_one, flow_neg_one)
                        #
                        total_loss += cfg.gradicon_weight * grad_icon
                        #
                        logging_dict.update({'grad_icon': grad_icon.detach().cpu()})

                    # interpolation flow consistency
                    with optional_context(cfg.interp_flow_weight == 0.0, torch.no_grad()):
                        # semi-group (longitudinal) consistency
                        # \phi_1 \circ \phi_(t-1) = \phi_t
                        flow_one_t_1 = compose(flow_one.expand(cfg.batch_size, *[-1] * 4), flow_t_1, cfg.compose_detach)
                        # \phi_-1 \circ \phi_t = \phi_(t-1)
                        flow_neg_one_t = compose(flow_neg_one.expand(cfg.batch_size, *[-1] * 4), flow_t,
                                                 cfg.compose_detach)
                        #
                        flow_interp = 0.5 * (
                                    loss_funcs['flow'](flow_one_t_1, flow_t, fg) +
                                    loss_funcs['flow'](flow_neg_one_t, flow_t_1, fg))
                        #
                        total_loss += cfg.interp_flow_weight * flow_interp
                        #
                        logging_dict.update({'flow_interp': flow_interp.detach().cpu()})
                    ########################################################################################

                    ########################### extrapolation constraint ###################################
                    # extrapolate in the negative direction
                    # target -> target_t_1 -> source
                    flow_21_0 = model(target_t_1.detach(), target.expand(cfg.batch_size, *[-1] * 4), 1.0 / (t - 1.0))
                    if vecint: flow_21_0 = vecint(flow_21_0)
                    ext_target = warp(target.expand(cfg.batch_size, *[-1] * 4), flow_21_0)

                    # extrapolate in the positive direction
                    # source -> source_t -> target
                    flow_01_2 = model(source.expand(cfg.batch_size, *[-1] * 4), source_t.detach(), 1.0 / t)
                    if vecint: flow_01_2 = vecint(flow_01_2)
                    ext_source = warp(source.expand(cfg.batch_size, *[-1] * 4), flow_01_2)

                    with optional_context(cfg.ext_sim_weight == 0.0, torch.no_grad()):
                        # image similarity
                        with torch.autocast(device_type=cfg.device, dtype=cfg.amp_dtype, enabled=False):
                            sim_ext = loss_funcs['sim'](
                                torch.cat([ext_target, ext_source], dim=0),
                                torch.cat([source.expand(cfg.batch_size, *[-1] * 4),
                                           target.expand(cfg.batch_size, *[-1] * 4)], dim=0)
                            )
                        #
                        total_loss += cfg.ext_sim_weight * sim_ext
                        #
                        logging_dict.update({'sim_ext': sim_ext.detach().cpu()})

                    with optional_context(cfg.ext_gradicon_weight == 0.0, torch.no_grad()):
                        # flow_21_0 warp target to source
                        # flow_01_2 warp source to target
                        # inverse consistency between flow_one (flow_01) and flow_21_0,
                        # and between flow_neg_one (flow_10) and flow_01_2
                        grad_icon_ext = loss_funcs['gradicon'](
                            torch.cat([flow_21_0, flow_01_2], dim=0),
                            torch.cat([flow_one.expand(cfg.batch_size, *[-1] * 4).detach(),
                                       flow_neg_one.expand(cfg.batch_size, *[-1] * 4).detach()], dim=0)
                        )
                        #
                        total_loss += cfg.ext_gradicon_weight * grad_icon_ext
                        #
                        logging_dict.update({'grad_icon_ext': grad_icon_ext.detach().cpu()})

                    with optional_context(cfg.ext_flow_weight == 0.0, torch.no_grad()):
                        # extrapolation flow supervision
                        flow_ext = 0.5 * (loss_funcs['flow'](flow_21_0,
                                                             flow_neg_one.expand(cfg.batch_size, *[-1] * 4).detach(),
                                                             fg)
                                          + loss_funcs['flow'](flow_01_2,
                                                               flow_one.expand(cfg.batch_size, *[-1] * 4).detach(),
                                                               fg))
                        # thresholding flow_ext to prevent instability
                        if flow_ext < 0.4:
                            total_loss += cfg.ext_flow_weight * flow_ext
                        else:
                            print(f'skipping flow_ext: {flow_ext.detach().cpu().item()}')
                        #
                        logging_dict.update({'flow_ext': flow_ext.detach().cpu()})

            logging_dict.update({'total_loss': total_loss.detach().cpu()})

            if phase == 'train':
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                # PSNR
                reg_psnr = metric_funcs['psnr'](y_source, target)
                # Jac det
                log_jacdet, non_pos_jacdet = metric_funcs['jacdet'](
                    flows.detach().cpu().numpy())

                logging_dict.update({
                    'reg_psnr': reg_psnr.detach().mean().cpu().item(),
                    'log_jacdet':
                        log_jacdet.mean(),
                    'non_pos_jacdet':
                        non_pos_jacdet.mean(),
                })

            # wandb logging
            wandb.log({phase: logging_dict})
