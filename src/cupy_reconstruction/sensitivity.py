from math import ceil
from typing import Union, Optional

from .utils import cartesian_bowl
from .sensitivity_cuda import calc_sensitivity

import numpy as np
import cupy as cp
from tqdm import tqdm


class SensitivityField:
    def __init__(self,
                 sound_speed_mm_per_s: float = 1525e3,
                 bandpass_freq_hz: tuple = (10e6, 120e6),
                 td_focal_length_mm=2.97,
                 td_diameter_mm=3.0,
                 td_focal_zone_factor: float = 1.02,
                 sampling_freq_hz=1e9,
                 n_sensor_points=4000,
                 precomputed_field=None,
                 precomputed_x=None,
                 precomputed_z=None):
        """
        Represents the sensitivity field of a focal transducer.

        [1] M. Schwartz. "Multispectral Optoacoustic Dermoscopy: Methods and Applications"
        (https://mediatum.ub.tum.de/1324031)

        Parameters
        ----------
        sound_speed_mm_per_s : float
            Assumed speeed of sound [mm/s]
        bandpass_freq_hz: tuple[float, float]
            Frequency range (f0, f1) of the bandpass filter [Hz]
        td_focal_length_mm: float
            Focal length of the transducer [mm]
        td_diameter_mm: float
            Diameter of the transducer [mm]
        td_focal_zone_factor: float
            Factor to determine the width of the focal zone [1, Formula 3.14]
        sampling_freq_hz : float
            Sampling frequency [Hz]
        n_sensor_points : int
            Number of evenly spaced integration points on the transducer surface
        precomputed_field : np.ndarray
            Precomputed sensitivity field
        precomputed_x : np.ndarray
            x-coordinates of the precomputed sensitivity field
        precomputed_z : np.ndarray
            z-coordinates of the precomputed sensitivity field
        """
        if precomputed_field is not None:
            self.is_precomputed = True

            assert precomputed_x is not None and precomputed_z is not None, "x and z must be provided when precomputed_field is given"
            assert precomputed_field.shape[0] == precomputed_z.shape[0], f"z-vector must be of shape {precomputed_field.shape[0]}, but is {precomputed_z.shape[0]}"
            assert precomputed_field.shape[1] == precomputed_x.shape[0], f"x-vector must be of shape {precomputed_field.shape[1]}, but is {precomputed_z.shape[0]}"

            self.field = precomputed_field
            self.x = precomputed_x
            self.z = precomputed_z

        else:
            self.is_precomputed = False

            self.sound_speed_mm_per_s = sound_speed_mm_per_s
            self.bandpass_freq_hz = bandpass_freq_hz
            self.td_focal_length_mm = td_focal_length_mm
            self.td_diameter_mm = td_diameter_mm
            self.td_focal_zone_factor = td_focal_zone_factor
            self.sampling_freq_hz = sampling_freq_hz
            self.n_sensor_points = n_sensor_points

            self.frequency_bands = np.array(
                [(bandpass_freq_hz[i], bandpass_freq_hz[i+1]) for i in range(len(bandpass_freq_hz) - 1)]
            )

            # hyperbolic focal zone
            # [1, Formula 3.13+3.14]
            td_focal_zone_width = td_focal_zone_factor * sound_speed_mm_per_s * td_focal_length_mm / (self.frequency_bands.mean(axis=1) * td_diameter_mm)  # [mm]
            self.hyper_a = td_focal_zone_width / 2
            self.hyper_b = (2 * td_focal_length_mm / td_diameter_mm) * self.hyper_a

            # model absorber signal
            # [1, Section: Sensitivity field simulation (p. 38+)]
            r_absorber = 0.8 * sound_speed_mm_per_s / self.frequency_bands.sum(axis=1)

            self.period = 1 / (2 * sampling_freq_hz)
            dt = sound_speed_mm_per_s * self.period  # time spacing
            signal_lengths = np.ceil(r_absorber / dt).astype(int) + 1

            # different frequency bands have different signal lengths, we chose the longest one to be able to
            # put them into one array
            max_length = signal_lengths.max()

            # we compute N-shaped signal without the unnecesary zeros at the end
            signal = [np.arange(length) * self.period * sound_speed_mm_per_s for length in signal_lengths]
            # pad all signals to the same length
            signal = np.array(
                [np.pad(sig, (0, max_length - len(sig))) for sig in signal]
            )
            # add the mirrored negative part to create the N-Shape
            self.signal = np.hstack(
                (-np.fliplr(signal[:, 1:]), signal)
            )

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
                 x: Optional[np.ndarray] = None, x_spacing: float = None,
                 max_vram_gb: Optional[Union[int, float]] = None, verbose=True):
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
            self.z + self.td_focal_length_mm,
            indexing='ij'
        ), axis=-1).squeeze()

        sensor_points = cartesian_bowl(
            self.td_focal_length_mm,
            self.td_diameter_mm,
            self.n_sensor_points
        )

        furthest_dist = np.linalg.norm(grid[-1, 0, None] - sensor_points, axis=-1) / self.sound_speed_mm_per_s
        min_index = np.floor(np.min(furthest_dist) / self.period)
        max_index = np.ceil(np.max(furthest_dist) / self.period)
        hist_size = int(max_index - min_index)

        grid = cp.asarray(grid, dtype=cp.float32)
        sensor_points = cp.asarray(sensor_points, dtype=cp.float32)
        signal = cp.asarray(self.signal, dtype=cp.float32)
        field = cp.zeros((len(self.signal), ) + grid.shape[:-1], dtype=np.float32)

        free_memory = cp.cuda.Device(0).mem_info[0]
        needed_memory = (grid.shape[0] * grid.shape[1] * hist_size * 4)

        # histogram = cp.zeros((480, 480, hist_size), dtype=cp.float32)

        if max_vram_gb is None and free_memory > needed_memory:
            histogram = cp.zeros((grid.shape[0], grid.shape[1], hist_size), dtype=cp.float32)
        else:
            if max_vram_gb is None:
                max_vram = free_memory
            else:
                max_vram = min(max_vram_gb * 1000 ** 3, free_memory)

            i = 0
            while 1:
                new_shape = (grid.shape[0] // (1 + i) + 1,  grid.shape[1] // (1 + i) + 1, hist_size)
                needed_memory = np.prod(new_shape) * 4
                if needed_memory < max_vram:
                    break
                i += 1
            histogram = cp.zeros(new_shape, dtype=cp.float32)

        calc_sensitivity(
            grid,
            sensor_points,
            histogram,
            field,
            signal,
            self.sound_speed_mm_per_s, self.period,
            verbose
        )

        self.field = field.get()
        self.field = np.swapaxes(self.field, 1, 2)
        self.field /= self.field.max(axis=(1, 2), keepdims=True)

        del grid, sensor_points, histogram, signal, field
        cp.get_default_memory_pool().free_all_blocks()

        self.field_width = (self.hyper_a / self.hyper_b * np.sqrt(self.hyper_b * self.hyper_b + np.abs(self.z)[:, None] ** 2)).T
