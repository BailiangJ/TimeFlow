# TimeFlow: Temporal Conditioning for Longitudinal Brain MRI Registration and Aging Analysis
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2501.08667-b31b1b.svg)](https://arxiv.org/abs/2501.08667)

keywords: Medical Image Registration, Longitudinal Analysis, Brain MRI, Deep Learning, PyTorch

This is a **PyTorch** implementation of our paper:

<a href="https://ieeexplore.ieee.org/document/11437527/">Jian, Bailiang, et al. "Temporal Conditioning for Longitudinal Brain MRI Registration and Aging Analysis." IEEE Transactions on Medical Imaging, 2026.</a>

## Architecture
<div align="center">
  <img src="figs/method_v1.png" alt="TimeFlow Architecture" width="95%">
</div>

## Installation

### Requirements
- Python >= 3.10
- PyTorch >= 2.1
- [MONAI](https://monai.io/) >= 1.3
- mmengine
- nibabel, scipy, pandas, numpy
- wandb (for experiment logging)
- ConfigArgParse

### Setup

```bash
git clone https://github.com/BailiangJ/TimeFlow.git
cd TimeFlow
pip install -r requirements.txt
```

## Data Preparation

TimeFlow assumes longitudinal brain MRI data preprocessed using the [FreeSurfer v7.2 longitudinal pipeline](https://surfer.nmr.mgh.harvard.edu/fswiki/LongitudinalProcessing).
The data should be structurally aligned and intensity normalized before passing to TimeFlow.

Your dataset information should be provided in:
- A flat directory containing all preprocessed images (e.g., `.mgz` or `.nii.gz` format).
- A `.csv` file detailing subject IDs, scan visits (e.g., ADNI dataframe).
- (Optional) a `.json` listing of subject IDs subsets to use.

See `scripts/train_cfg.py` for variables mapping to the data structures.

## Training

### Configuration
All hyperparameters and model settings reside in `scripts/train_cfg.py`. Key parameters include:
- `interp_flow_weight`: the weight of the proposed interpolation flow consistency constraint.
- `ext_sim_weight`: the weight of extrapolation consistency constraint.
- `ext_flow_weight`: the weight of extrapolation flow consistency constraint.

### Run Training
To launch training with tracking, point to the configuration file:
```bash
python scripts/train.py --train-config scripts/train_cfg.py --random-seed 42
```

## Inference

To use a trained TimeFlow model to predict deformed brain scans or deformation fields:
```bash
python scripts/infer.py -m output_model_folder -exp <exp_id> -epoch <epoch_id>
```

In `scripts/infer_cfg.py`, you can change prediction settings, including the source and target evaluation times.

## Pretrained Weights
TBD

## Acknowledgments
We would like to acknowledge the following excellent repositories that our codebase builds upon:
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
- [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
- [GradICON](https://github.com/uncbiag/icon)


## Citation

If you use TimeFlow in your research, please consider citing our work:

```bibtex
@article{jian2026temporal,
  title={Temporal Conditioning for Longitudinal Brain MRI Registration and Aging Analysis},
  author={Jian, Bailiang and Pan, Jiazhen and Li, Yitong and Bongratz, Fabian and Li, Ruochen and Rueckert, Daniel and Wiestler, Benedikt and Wachinger, Christian},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
```

## License
© Bailiang Jian
Licensed under the [MIT Licensce](LICENSCE)