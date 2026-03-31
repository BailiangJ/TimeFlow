from monai.metrics import (DiceMetric, HausdorffDistanceMetric,
                           SurfaceDistanceMetric)

from ..builder import METRICS
from .sdlogjac import SDlogDetJac, compute_jacdet_map
from .fg_sdlogjac import Fg_SDlogDetJac
from .psnr import FgPSNR

METRICS.register_module('dice', module=DiceMetric)
METRICS.register_module('haus_dist', module=HausdorffDistanceMetric)
METRICS.register_module('surf_dist', module=SurfaceDistanceMetric)

__all__ = ['SDlogDetJac', 'Fg_SDlogDetJac', 'FgPSNR', 'compute_jacdet_map']
