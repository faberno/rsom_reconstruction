import cupy as cp
import numpy as np

def recon2rgb(recon, no_blue=False):
    xp = cp.get_array_module(recon)
    recon /= recon.max(axis=tuple(range(1, recon.ndim)), keepdims=True)
    recon = xp.clip(recon, a_min=0, a_max=1)
    if not no_blue:
        recon = xp.vstack((recon, np.zeros((1, ) + recon.shape[1:])))
    recon = xp.moveaxis(recon, 0, -1)

    return recon