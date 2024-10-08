from math import ceil
from typing import Union, Optional

from .utils import make_cart_bowl
from .sensitivity_cuda import calc_sensitivity

import numpy as np
import cupy as cp
from tqdm import tqdm


class SensitivityField:
    def __init__(self,
                 sos: float = 1525e3,  # [mm/s]
                 f_bandpass: tuple = (15e6, 42e6, 120e6),  # [Hz]
                 td_focal_length=2.97,  # [mm]
                 td_diameter=3.0,  # [mm]
                 td_focal_zone_factor: float = 1.02,
                 fs=1e9,  # [Hz]
                 N_sensor_points=4000,
                 clip_method: str = 'cutoff',
                 cutoff: Optional[float] = 0.01,
                 precomputed_field=None,
                 precomputed_x=None,
                 precomputed_z=None):
        """
        Initialized the sensitivity field of a focal transducer.

        [1] M. Schwartz. "Multispectral Optoacoustic Dermoscopy: Methods and Applications"
        (https://mediatum.ub.tum.de/1324031)

        Parameters
        ----------
        sos : float
            Assumed speeed of sound [mm/s]
        f_bandpass: tuple[float, float]
            Frequency range (f0, f1) of the bandpass filter [Hz]
        td_focal_length: float
            Focal length of the transducer [mm]
        td_diameter: float
            Diameter of the transducer [mm]
        td_focal_zone_factor: float
            Factor to determine the width of the focal zone [1, Formula 3.14]
        fs : float
            Sampling frequency [Hz]
        N_sensor_points : int
            Number of evenly spaced integration points on the transducer surface
        cutoff: float
            Threshold at which values of the sensitivity field are set to zero
        precomputed_field : np.ndarray
            Precomputed sensitivity field
        precomputed_x : np.ndarray
            x-coordinates of the precomputed sensitivity field
        precomputed_z : np.ndarray
            z-coordinates of the precomputed sensitivity field
        """
        if precomputed_field is not None:
            self.is_precomputed = True

            assert precomputed_field.shape[0] == precomputed_z.shape[0]
            assert precomputed_field.shape[1] == precomputed_x.shape[0]

            self.field = precomputed_field
            self.x = precomputed_x
            self.z = precomputed_z

        else:
            self.is_precomputed = False

            self.sos = sos
            self.f_bandpass = f_bandpass
            self.td_focal_length = td_focal_length
            self.td_diameter = td_diameter
            self.td_focal_zone_factor = td_focal_zone_factor
            self.fs = fs
            self.N_sensor_points = N_sensor_points
            self.clip_method = clip_method
            self.cutoff = cutoff

            self.bands = np.array([(f_bandpass[i], f_bandpass[i+1]) for i in range(len(f_bandpass) - 1)])

            # sensitivity field hyperbola
            td_focal_zone_width = td_focal_zone_factor * sos * td_focal_length / (self.bands.mean(axis=1) * td_diameter)  # [mm]
            self.hyper_a = td_focal_zone_width / 2
            self.hyper_b = (2 * td_focal_length / td_diameter) * self.hyper_a

            # model absorber signal [1, Section: Sensitivity field simulation (p. 38+)]
            r_absorber = 0.8 * sos / self.bands.sum(axis=1)

            # create time vector
            self.period = 1 / (2 * fs)
            spacing = sos * self.period
            signal_length = np.ceil(r_absorber / spacing).astype(int) + 1
            max_length = signal_length.max()

            self.signal = [np.arange(length) * self.period * sos for length in signal_length]  # we compute N-shaped signal without the unnecesary zeros at the end
            self.signal = [np.hstack((signal, np.zeros(max_length - len(signal)))) for signal in self.signal]  # pad the smaller signal to get same lengths
            self.signal = np.array(self.signal)
            self.signal = np.hstack((-np.fliplr(self.signal[:, 1:]), self.signal))  # add the mirrored negative part for N-Shape

    @staticmethod
    def from_precomputed(field, x, z):
        sensitivity_field = SensitivityField(
            precomputed_field=field,
            precomputed_x=x,
            precomputed_z=z
        )
        return sensitivity_field

    def simulate(self,
                 z: np.ndarray,
                 x: Optional[np.ndarray] = None, x_spacing: float = None):
        """
        Simulates the sensitivity field of a focal transducer.

        [1] M. Schwartz. "Multispectral Optoacoustic Dermoscopy: Methods and Applications"
        (https://mediatum.ub.tum.de/1324031)

        Parameters
        ----------
        z : np.ndarray
            z-coordinates of the sensitivity field
        x : np.ndarray
            x-coordinates of the sensitivity field. If not provided, the max radius of
            the hyperbola is used as furthest point in x-direction
        x_spacing : float
            Spacing of the x-coordinates if x is not provided
        """
        if self.is_precomputed:
            raise ValueError("Sensitivity field is precomputed and cannot be simulated")
        if x is None and x_spacing is None:
            raise ValueError("Either x or x_spacing must be provided")
        if x is None:
            max_depth = np.abs(z).max()
            r_max = (self.hyper_a / self.hyper_b * np.sqrt(self.hyper_b * self.hyper_b + max_depth ** 2)).max()
            n_x = int(ceil(r_max.item() / x_spacing)) + 1
            self.x = np.arange(0, n_x) * x_spacing
        else:
            self.x = np.asarray(x)

        self.z = np.asarray(z)

        grid = np.stack(np.meshgrid(
            self.x,
            0,
            self.z + self.td_focal_length,
            indexing='ij'
        ), axis=-1).squeeze()

        sensor_points = make_cart_bowl(
            np.array([0, 0, 0]),
            self.td_focal_length,
            self.td_diameter,
            np.array([0, 0, 1]),
            self.N_sensor_points).T

        furthest_dist = np.linalg.norm(grid[-1, 0, None] - sensor_points, axis=-1) / self.sos
        min_index = np.floor(np.min(furthest_dist) / self.period)
        max_index = np.ceil(np.max(furthest_dist) / self.period)
        hist_size = int(max_index - min_index)

        histogram = cp.zeros((512, 640, hist_size), dtype=cp.float32)
        field = cp.zeros((len(self.signal), ) + grid.shape[:-1], dtype=np.float32)

        grid = cp.asarray(grid, dtype=cp.float32)
        sensor_points = cp.asarray(sensor_points, dtype=cp.float32)
        signal = cp.asarray(self.signal, dtype=cp.float32)


        calc_sensitivity(
            grid,
            sensor_points,
            histogram,
            field,
            signal,
            self.sos, self.period
        )

        self.field = field.get()
        self.field /= self.field.max(axis=(1, 2), keepdims=True)

        if self.clip_method == 'cutoff':
            mask = self.field < self.cutoff
            self.field_width = np.argmax(mask, axis=1) * x_spacing  # todo: x_spacing not defined when given x
        elif self.clip_method == 'hyperbola':
            self.field_width = (self.hyper_a / self.hyper_b * np.sqrt(self.hyper_b * self.hyper_b + np.abs(self.z)[:, None] ** 2)).T
        else:
            self.field_width = np.ones(self.field.shape[0]) * self.x[-1]

    # @staticmethod
    # def get_sensitivity(pos, sensor_points, sos, period, signal):
    #     dist = np.linalg.norm(sensor_points - pos, axis=-1)
    #     tof = dist / sos
    #
    #     # start and end of histogram
    #     min_index = np.floor(np.min(tof) / period)
    #
    #     bin = (tof / period) - min_index
    #     bin_ceil = np.floor(bin).astype(int) + 1
    #     ceil_fractions = bin - (bin_ceil - 1)
    #
    #     weights = 1 / (2 * np.pi * dist)
    #     ceil_weight = ceil_fractions * weights
    #     floor_weights = (1 - ceil_fractions) * weights
    #
    #     total = np.bincount(bin_ceil, weights=ceil_weight)
    #     total += np.bincount(bin_ceil - 1, weights=floor_weights, minlength=len(total))
    #
    #     nShapeConv = np.convolve(signal, total)
    #     return np.max(nShapeConv) - np.min(nShapeConv)

    # def simulate_cpu(self,
    #              z: np.ndarray,
    #              x: Optional[np.ndarray] = None, x_spacing: float = None):
    #     """
    #     Simulates the sensitivity field of a focal transducer.
    #
    #     [1] M. Schwartz. "Multispectral Optoacoustic Dermoscopy: Methods and Applications"
    #     (https://mediatum.ub.tum.de/1324031)
    #
    #     Parameters
    #     ----------
    #     z : np.ndarray
    #         z-coordinates of the sensitivity field
    #     x : np.ndarray
    #         x-coordinates of the sensitivity field. If not provided, the max radius of
    #         the hyperbola is used as furthest point in x-direction
    #     x_spacing : float
    #         Spacing of the x-coordinates if x is not provided
    #     """
    #     if self.is_precomputed:
    #         raise ValueError("Sensitivity field is precomputed and cannot be simulated")
    #     if x is None and x_spacing is None:
    #         raise ValueError("Either x or x_spacing must be provided")
    #     if x is None:
    #         max_depth = np.abs(z).max()
    #         r_max = self.hyper_a / self.hyper_b * sqrt(self.hyper_b * self.hyper_b + max_depth ** 2)
    #         self.x = np.arange(0, r_max + x_spacing, x_spacing, dtype=np.float32)
    #     else:
    #         self.x = np.asarray(x)
    #
    #     self.z = np.asarray(z)
    #
    #     grid = np.stack(np.meshgrid(
    #         self.x,
    #         0,
    #         self.z + self.td_focal_length,
    #         indexing='ij'
    #     ), axis=-1)
    #     shape = grid.shape[:-1]
    #     grid_points = grid.reshape(-1, 3)
    #
    #     sensor_points = make_cart_bowl(
    #         np.array([0, 0, 0]),
    #         self.td_focal_length,
    #         self.td_diameter,
    #         np.array([0, 0, 1]),
    #         self.N_sensor_points).T
    #
    #     with Pool() as pool:
    #         result = pool.starmap(SensitivityField.get_sensitivity, tqdm(((grid_points[i], sensor_points, self.sos, self.period, self.signal) for i in range(len(grid_points))), total=len(grid_points)))
    #     self.field = np.asarray(result).reshape(shape).squeeze().T
    #     self.field = self.field / self.field.max()
    #
    #     if self.cutoff is not None:
    #         mask = self.field < self.cutoff
    #         self.field[mask] = 0
    #         self.field_width = np.argmax(mask, axis=1) * x_spacing  # todo: x_spacing not defined when given x
    #     else:
    #         self.field_width = np.ones(self.field.shape[0]) * self.x[-1]
