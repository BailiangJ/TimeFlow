import glob
import nibabel as nib
import json
import numpy as np
import os
import pandas as pd
import sys
import torch
from copy import deepcopy
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (CastToTyped, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandSpatialCropSamplesd, Resized,
                              ResizeWithPadOrCropd, ScaleIntensityd,
                              ScaleIntensityRangePercentilesd, Spacingd,
                              ToTensord)
from monai.utils import first, set_determinism
from typing import (Any, Callable, Dict, Hashable, List, Mapping, Optional,
                    Sequence, Tuple, Union)
from .data_transform import ScaleIntensityRanged
from models import CFG


def load_data_01(cfg: CFG,
              *args, **kwargs):
    data_dicts = []
    df = pd.read_csv(cfg.adni_df)
    df = df.sort_values(['PTID', 'Month'])
    with open(cfg.subset_json) as f:
        rids = json.load(f)
    for p in rids:
        d_dict = {}
        subject_df = df.loc[df['PTID'] == p]
        num_tps = len(subject_df)
        image_uid_0 = subject_df.iloc[0]['IMAGEUID'] # baseline visit
        image_uid_1 = subject_df.iloc[num_tps - 1]['IMAGEUID'] # endpoint of the sequence
        scan_path_0 = os.path.join(cfg.data_dir,
                                   f'{image_uid_0}.long.{p}/mri/norm.mgz')
        scan_path_1 = os.path.join(cfg.data_dir,
                                   f'{image_uid_1}.long.{p}/mri/norm.mgz')
        d_dict.update({
            'subject': p,
            't0_vis': subject_df.iloc[0]['VISCODE'],
            't0_uid': image_uid_0,
            't0': scan_path_0,
            't1_vis': subject_df.iloc[num_tps - 1]['VISCODE'],
            't1_uid': image_uid_1,
            't1': scan_path_1,
            'num_tps': num_tps,
        })

        data_dicts.append(d_dict)

    print(len(data_dicts))
    data_dicts = data_dicts[slice(*cfg.dataset_slice)]

    scan_keys = ['t0', 't1']

    pre_transforms = [
        LoadImaged(keys=scan_keys, reader='NibabelReader', image_only=True),
        EnsureChannelFirstd(keys=scan_keys,
                            channel_dim='no_channel'),
        # ADNI long all oriented LIA
        # Orientationd(keys=scan_keys + seg_keys, axcodes='LIA'),
        ScaleIntensityRanged(keys=scan_keys,
                             a_min=0.0,
                             upper=99.9,
                             b_min=0.0,
                             b_max=1.0,
                             clip=False),
    ]
    crop_fg_transforms = [
        CropForegroundd(keys=scan_keys,
                        source_key='t0',
                        k_divisible=2)
    ]
    post_transforms = [
        ResizeWithPadOrCropd(keys=scan_keys,
                             spatial_size=cfg.image_size,
                             mode='constant'),
        # OneHotd(keys=seg_keys),
        ToTensord(keys=scan_keys, track_meta=False)
    ]

    data_transforms = Compose(pre_transforms + crop_fg_transforms +
                              post_transforms)

    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset


def load_data_tps(cfg: CFG,
                       subject_df: pd.DataFrame,
                       *args,
                       **kwargs):
    rids = subject_df['PTID'].unique()
    assert len(rids)==1
    p = rids[0]
    num_timepoints = len(subject_df)

    d_dict = {}
    for i in range(num_timepoints):
        image_uid = subject_df.iloc[i]['IMAGEUID']
        scan_path = os.path.join(cfg.data_dir,
                                    f'{image_uid}.long.{p}/mri/norm.mgz')
        d_dict.update({
            'subject': p,
            f't{i}_vis': subject_df.iloc[i]['VISCODE'],
            f't{i}_uid': image_uid,
            f't{i}': scan_path,
            f't{i}_years_bl': subject_df.iloc[i]['Years_bl'],
        })
    data_dicts = [d_dict]

    print(f'tp{num_timepoints}:', len(data_dicts))
    scan_keys = [f't{i}' for i in range(num_timepoints)]

    if cfg.image_size == [256] * 3: # no cropping
        print('loading uncropped data.')
        data_transforms = Compose([
            LoadImaged(keys=scan_keys, reader='NibabelReader', image_only=True),
            EnsureChannelFirstd(keys=scan_keys,
                                channel_dim='no_channel'),
            # ADNI long all oriented LIA
            # Orientationd(keys=scan_keys + seg_keys, axcodes='LIA'),
            ScaleIntensityRanged(keys=scan_keys,
                                 a_min=0.0,
                                 upper=99.9,
                                 b_min=0.0,
                                 b_max=1.0,
                                 clip=False),
            ToTensord(keys=scan_keys,
                      track_meta=False)
        ])
    else:
        data_transforms = Compose([
            LoadImaged(keys=scan_keys, reader='NibabelReader', image_only=True),
            EnsureChannelFirstd(keys=scan_keys,
                                channel_dim='no_channel'),
            # ADNI long all oriented LIA
            # Orientationd(keys=scan_keys, axcodes='LIA'),
            ScaleIntensityRanged(keys=scan_keys,
                                 a_min=0.0,
                                 upper=99.9,
                                 b_min=0.0,
                                 b_max=1.0,
                                 clip=False),
            CropForegroundd(keys=scan_keys,
                            source_key='t0',
                            k_divisible=2),
            ResizeWithPadOrCropd(keys=scan_keys,
                                 spatial_size=cfg.image_size,
                                 mode='constant'),
            ToTensord(keys=scan_keys, track_meta=False)
        ])

    dataset = CacheDataset(data=data_dicts,
                           transform=data_transforms,
                           *args,
                           **kwargs)
    return dataset