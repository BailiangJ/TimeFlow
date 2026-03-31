import torch

device = 'cuda'
amp_dtype = 'bfloat16' if device == 'cpu' else 'float16'
use_amp = True
image_size = [160, 160, 192]

# wandb
project = 'TimeFlow'
group = 'timeflow'
name = 'tf-10interp+1simext+.1extgd+1e-3flowext'

exp_id = 0
# output directory
out_path = f'./tf_outputs/exp{exp_id}'
model_dir = 'saved_models'

# load model
load_model = None

# TODO: fill in the paths for the dataset and 
data_dir = '/data/images/'
adni_df = ''
subset_json = ''

load_01_data = True
batch_size = 2

cache_rate = 1
num_workers = -1
dataset_slice = (None, None, 1)

oversample_rate = 0.0
compose_detach = True

# pairwise endpoint registration
sim_weight = 1.0
gradicon_weight = 0.0 # oversample_rate is fixed to 0 therefore gradicon_weight is not set

# interpolation consistency constraint
interp_flow_weight = 10.0

# extrapolation consistency constraints
ext_sim_weight = 1.0
ext_gradicon_weight = 0.1
ext_flow_weight = 1e-3

# optimizer and lr_scheduler
lr = 1e-4
lr_decay = 0.999
start_epoch = 0
max_epochs = 50
save_interval = 20

# common unsupervised losses
# similarity loss
# sim_loss_cfg = dict(type='IntensityLoss', penalty='l2', weight=sim_weight)
sim_loss_cfg = dict(type='ncc',
                    spatial_dims=3,
                    kernel_size=9,
                    smooth_nr=0.0,
                    smooth_dr=1e-5,
                    weight=sim_weight)

# smoothness regularization
reg_loss_cfg = dict(type='diffusion',
                    penalty='l2',
                    loss_mult=1.0,
                    weight=0.0)

# flow losses
flow_loss_cfg = dict(type='FlowLoss',
                     penalty='l2')

# gradICON loss
gradicon_loss_cfg = dict(type='GradICONLoss',
                         flow_loss_cfg=flow_loss_cfg,
                         image_size=image_size,
                         interp_mode='bilinear',
                         compose_detach=False,
                         delta=1e-3,
                         weight=gradicon_weight,
                         )

# velocity field integration
if 'diff' in group:
    vecint_cfg = dict(
        type='VecIntegrate',
        image_size=image_size,
        num_steps=7,
        interp_mode='bilinear',
    )
else:
    vecint_cfg = None

# registration head
registration_cfg = dict(type='RegistrationHead',
                        image_size=image_size,
                        spatial_scale=1.0,
                        flow_scale=1.0,
                        interp_mode='bilinear')

t_embed_dim = 16
adaptive_norm = True
model_cfg = dict(
    type='TimeFlow',
    t_embed_dim=t_embed_dim,
    pe_type='spe',
    max_periods=100,
    encoder_cfg=dict(
        spatial_dims=3,
        in_chan=2,
        down=True,
        out_channels=[32, 32, 48, 48, 96],
        out_indices=[0, 1, 2, 3, 4],
        block_config=dict(
            kernel_size=3,
            t_embed_dim=t_embed_dim,
            adaptive_norm=adaptive_norm,
            down_first=True,
            conv_down=True,
            bias=True,
            norm_name=('INSTANCE', {'affine': False if adaptive_norm else True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
            dropout=None,
        ),
    ),
    decoder_cfg=dict(
        spatial_dims=3,
        skip_channels=[96, 48, 48, 32, 32],
        out_channels=[96, 48, 48, 32, 32],
        block_config=dict(
            kernel_size=3,
            t_embed_dim=t_embed_dim,
            adaptive_norm=adaptive_norm,
            up_transp_conv=True,
            transp_bias=False,
            upsample_kernel_size=2,
            bias=True,
            norm_name=('INSTANCE', {'affine': False if adaptive_norm else True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
            dropout=None,
        ),
    ),
    remain_cfg=dict(
        spatial_dims=3,
        in_chan=32,
        down=False,
        out_channels=[32] * 2,
        out_indices=[0],
        block_config=dict(
            kernel_size=3,
            t_embed_dim=t_embed_dim,
            adaptive_norm=adaptive_norm,
            bias=True,
            norm_name=('INSTANCE', {'affine': False if adaptive_norm else True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
            dropout=None,
        ),
    )
)
