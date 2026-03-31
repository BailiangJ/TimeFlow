import os
import sys
import signal
import random
from contextlib import contextmanager
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .data_transform import ScaleIntensityRanged
from .data_utils import load_data_01, load_data_tps
from .digital_diffeomorphism import calc_jac_dets, calc_measurements, get_identity_grid

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')


def worker_init_fn(worker_id):
    """Check https://github.com/Project-MONAI/MONAI/issues/1068."""
    worker_info = torch.utils.data.get_worker_info()
    try:
        worker_info.dataset.transform.set_random_state(worker_info.seed %
                                                       (2 ** 32))
    except AttributeError:
        pass


@contextmanager
def optional_context(condition, context_manager):
    if condition:
        with context_manager:
            yield
    else:
        yield


def save_checkpoint(epoch, save_dir, model, optimizer, lr_scheduler, scaler):
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch:04d}.pth'))
    state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict()
    }
    torch.save(state, os.path.join(save_dir, f'checkpoint_{epoch:04d}.pth'))
    print(f"Emergency model save at epoch {epoch}")


# Signal handler function
def register_signal_handler(epoch_func, save_dir, model, optimizer, lr_scheduler, scaler):
    # Termination signal handler
    def signal_handler(signum, frame):
        print("Termination signal received. Saving model before exit.")
        save_checkpoint(epoch_func(), save_dir, model, optimizer, lr_scheduler, scaler)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)