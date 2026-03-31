import os
import sys
import pandas as pd
import json
import numpy as np
import torch
from monai.data import DataLoader
from monai.metrics import SSIMMetric
from monai.data.utils import first
from models import build_metrics, build_registration_head, build_flow_estimator, Warp
from mmengine import Config, ConfigDict
import torch.nn as nn
from utils import load_data_tps as load_data
from utils import get_identity_grid, calc_jac_dets, calc_measurements

torch.backends.cudnn.deterministic = True

def infer(cfg):
    model = build_flow_estimator(cfg.model_cfg)
    if cfg.load_model is not None:
        model.load_state_dict(
            torch.load(cfg.load_model, map_location=torch.device(cfg.device)))
    model.to(cfg.device)
    model.eval()

    # build registration head module
    reg_head = build_registration_head(cfg.registration_cfg)
    reg_head.to(cfg.device)

    # warping layer
    warp = Warp(image_size=cfg.registration_cfg.image_size,
                interp_mode=cfg.registration_cfg.interp_mode)
    warp.to(cfg.device)

    # metrics:
    # SSIM, PSNR, NDV, logJacDet
    compute_ssim = SSIMMetric(data_range=torch.tensor(1.0,
                                                      dtype=torch.float32,
                                                      device=cfg.device),
                              spatial_dims=3,
                              reduction='none')._compute_metric
    compute_psnr = build_metrics(dict(type='fg_psnr', max_val=1.0))._compute_metric
    compute_jacdet = build_metrics(dict(type='fg_sdlogjac'))
    # get_identity_grid((3,H,W,D)) -> voxel-wise grid
    # calc_jac_dets(deformation)
    # calc_measurements(jac_dets, mask(image>0))
    id_grid = get_identity_grid(np.empty((3, *cfg.image_size)))
    metric_funcs = dict(ssim=compute_ssim,
                        psnr=compute_psnr,
                        jacdet=compute_jacdet)

    df = pd.read_csv(cfg.adni_df)
    subject_df = df.loc[df['PTID'] == cfg.rid]
    subject_df.sort_values(['Yars_bl'])
    subject_df.reset_index(drop=True, inplace=True)
    print(subject_df)
    dataset = load_data(cfg, subject_df)
    dataloader = DataLoader(dataset)

    with torch.no_grad():
        data = first(dataloader)
        source = data[f't{cfg.src_tp}'].float().to(cfg.device)
        target = data[f't{cfg.tgt_tp}'].float().to(cfg.device)

        # t \in [0, 2]
        # interpolation: 0<t<1
        # pairwise registration: t=1
        # extrapolation: t>1
        t_list = torch.arange(0.0, 2.0, 0.1).to(cfg.device)
        for t in t_list:
            t = t.unsqueeze(0)
            flow_t = model(source, target, t)
            if 'SVF' in cfg.registration_cfg.type:
                flow, _, y_src, _, _, _ = reg_head(flow, source)


if __name__ == '__main__':
    import pathlib
    import configargparse
    from utils import set_seed

    p = configargparse.ArgParser()
    p.add_argument('--method-folder',
                   '-m',
                   required=True,
                   type=lambda f: pathlib.Path(f).absolute(),
                   help='path of method folder')
    p.add_argument('--exp-id',
                   '-exp',
                   required=True,
                   type=int)
    p.add_argument('--epoch-id',
                   '-epoch',
                   required=True,
                   type=int)
    args = p.parse_args()
    save_dir = os.path.join(args.method_folder, f'exp{args.exp_id}')
    load_model = os.path.join(args.method_folder, f'exp{args.exp_id}/saved_models/{args.epoch_id:04d}.pth')
    train_cfg = Config.fromfile(os.path.join(args.method_folder, f'exp{args.exp_id}/train_configs.py'))
    config = Config.fromfile('./infer_cfg.py')

    if train_cfg.vecint_cfg is not None:
        registration_cfg = ConfigDict(type='SVFIntegrateHead',
                                      image_size=config.image_size,
                                      int_steps=7,
                                      resize_scale=1,
                                      resize_first=False,
                                      bidir=False,
                                      interp_mode='bilinear')
    else:
        registration_cfg = ConfigDict(type='RegistrationHead',
                                      image_size=config.image_size,
                                      spatial_scale=1.0,
                                      flow_scale=1.0,
                                      interp_mode='bilinear')

    config.update(dict(
        model_cfg=train_cfg.model_cfg,
        save_dir=save_dir,
        load_model=load_model,
        registration_cfg=registration_cfg,
        epoch_id=args.epoch_id,
    ))
    set_seed(2023)
    infer(config)