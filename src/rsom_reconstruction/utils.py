from math import cos, pi, sqrt, asin
from pathlib import Path
import threading
from typing import Union, Optional

import cupy as cp
import numpy as np
from scipy.io import savemat

ndarray = Union[np.ndarray, cp.ndarray]


def cartesian_bowl(focal_length_mm: float,
                   diameter_mm: float,
                   n_points: int) -> np.ndarray:
    """Create evenly spaced points on a spherical bowl.

    Simplified code of https://github.com/waltsims/k-wave-python/blob/master/kwave/utils/mapgen.py: make_cart_bowl
    """
    GOLDEN_ANGLE = pi * (3 - sqrt(5.))  # golden angle in radians

    # compute arc angle from chord (ref: https://en.wikipedia.org/wiki/Chord_(geometry))
    varphi_max = asin(diameter_mm / (2 * focal_length_mm))

    # spiral parameters
    t = np.arange(n_points)
    theta = GOLDEN_ANGLE * t
    C = 2 * np.pi * (1 - cos(varphi_max)) / (n_points - 1)
    varphi = np.arccos(1 - C * t / (2 * np.pi))

    # compute canonical spiral points
    p0 = np.array([
        np.cos(theta) * np.sin(varphi),
        np.sin(theta) * np.sin(varphi),
        np.cos(varphi)
    ]).T
    p0 = focal_length_mm * p0

    bowl = -p0 + np.array([0, 0, focal_length_mm])

    return bowl


def write_to_matfile(file_path: Union[str, Path], recon: ndarray, split_channels: bool = True, compress: bool = False):
    """
    Export the reconstruction to a mat file.

    Parameters
    ----------
    file_path: str, Path
        Path to the mat file
    recon: ndarray
        Reconstruction to be saved
    split_channels: bool
        If True, the frequency channels of the reconstruction are saved separately.
    compress: bool
        If True, the mat file is compressed
    """
    if not isinstance(recon, np.ndarray):
        recon = recon.get()

    if isinstance(file_path, Path):
        file_path = str(file_path)

    if not file_path.endswith('.mat'):
        file_path += '.mat'

    if not split_channels:
        out_dict = {'R': recon}
        save_thread = threading.Thread(target=savemat, args=(file_path, out_dict))
        save_thread.start()
    else:
        if len(recon) == 2:
            channel_names = ['LF', 'HF']
        else:
            if len(recon) > 4:
                raise ValueError("Too many channels to save. The 0th axis should describe the frequency channels.")
            channel_names = [f'C{c}' for c in range(len(recon))]
        for i, channel_name in enumerate(channel_names):
            out_dict = {'R': recon[i]}
            save_thread = threading.Thread(
                target=savemat,
                args=(file_path.replace(".mat", f"_{channel_name}.mat"), out_dict),
                kwargs={'do_compression': compress}
            )
            save_thread.start()
