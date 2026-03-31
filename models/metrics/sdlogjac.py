import numpy as np
import scipy

from ..builder import METRICS


def compute_jacdet_map(disp: np.ndarray):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([
        scipy.ndimage.correlate(
            disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
        scipy.ndimage.correlate(
            disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
        scipy.ndimage.correlate(
            disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)
    ],
        axis=1)

    # grady_disp: (B, 3, H, W, D)
    grady_disp = np.stack([
        scipy.ndimage.correlate(
            disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
        scipy.ndimage.correlate(
            disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
        scipy.ndimage.correlate(
            disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)
    ],
        axis=1)

    # gradz_disp: (B, 3, H, W, D)
    gradz_disp = np.stack([
        scipy.ndimage.correlate(
            disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
        scipy.ndimage.correlate(
            disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
        scipy.ndimage.correlate(
            disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)
    ],
        axis=1)

    # grad_disp: (B, 3, 3, H, W, D)
    grad_disp = np.stack([gradx_disp, grady_disp, gradz_disp], 1)

    # jacobian: (B, 3, 3, H, W, D)
    # displacement jacobian to deformation jacobian
    jacobian = grad_disp + np.eye(3, 3).reshape(1, 3, 3, 1, 1, 1)
    # jacdet: (B, H, W, D)
    jacdet = jacobian[:, 0, 0, ...] * \
             (jacobian[:, 1, 1, ...] * jacobian[:, 2, 2, ...] -
              jacobian[:, 1, 2, ...] * jacobian[:, 2, 1, ...]) - \
             jacobian[:, 1, 0, ...] * \
             (jacobian[:, 0, 1, ...] * jacobian[:, 2, 2, ...] -
              jacobian[:, 0, 2, ...] * jacobian[:, 2, 1, ...]) + \
             jacobian[:, 2, 0, ...] * \
             (jacobian[:, 0, 1, ...] * jacobian[:, 1, 2, ...] -
              jacobian[:, 0, 2, ...] * jacobian[:, 1, 1, ...])
    return jacdet


@METRICS.register_module('sdlogjac')
class SDlogDetJac:
    def __call__(self, disp: np.ndarray, ):
        '''
        Args:
            disp: displacement field of shape (B, 3, H, W, D)
        '''
        _, _, H, W, D = disp.shape

        gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
        grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
        gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

        # Compute the gradient of the displacement field
        # gradx_disp: (B, 3, H, W, D)
        gradx_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)
        ],
            axis=1)

        # grady_disp: (B, 3, H, W, D)
        grady_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)
        ],
            axis=1)

        # gradz_disp: (B, 3, H, W, D)
        gradz_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)
        ],
            axis=1)

        # grad_disp: (B, 3, 3, H, W, D)
        grad_disp = np.stack([gradx_disp, grady_disp, gradz_disp], 1)

        # jacobian: (B, 3, 3, H, W, D)
        # displacement jacobian to deformation jacobian
        jacobian = grad_disp + np.eye(3, 3).reshape(1, 3, 3, 1, 1, 1)
        jacobian = jacobian[:, :, :, 2:-2, 2:-2, 2:-2]
        # jacdet: (B, H, W, D)
        jacdet = jacobian[:, 0, 0, ...] * \
                 (jacobian[:, 1, 1, ...] * jacobian[:, 2, 2, ...] -
                  jacobian[:, 1, 2, ...] * jacobian[:, 2, 1, ...]) - \
                 jacobian[:, 1, 0, ...] * \
                 (jacobian[:, 0, 1, ...] * jacobian[:, 2, 2, ...] -
                  jacobian[:, 0, 2, ...] * jacobian[:, 2, 1, ...]) + \
                 jacobian[:, 2, 0, ...] * \
                 (jacobian[:, 0, 1, ...] * jacobian[:, 1, 2, ...] -
                  jacobian[:, 0, 2, ...] * jacobian[:, 1, 1, ...])

        non_pos_jacdet = np.sum(jacdet <= 0, axis=(1, 2, 3))

        jacdet = (jacdet + 3).clip(0.000000001, 1000000000)
        log_jacdet = np.log(jacdet)

        return np.std(log_jacdet, axis=(1, 2, 3)), non_pos_jacdet
