from math import cos, pi, sqrt, asin
from typing import Union, Optional

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ndarray = Union[np.ndarray, cp.ndarray]


def cartesian_bowl(focal_length_mm: float,
                   diameter_mm: float,
                   n_points: int) -> np.ndarray:
    """Create evenly spaced points on a spherical bowl.

    Code simplification of
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
