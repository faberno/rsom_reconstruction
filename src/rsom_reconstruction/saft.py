from pathlib import Path
from typing import Optional, Union

import cupy as cp
import hdf5storage as h5
import numpy as np

from .preprocessing import preprocess_signal
from .saft_cuda import run_saft
from .sensitivity import SensitivityField
from .utils import ndarray


def saft_matfile_adapter(signal_path: Union[str, Path],
                         sensitivity_field: SensitivityField,
                         reconstruction_grid_spacing_mm: tuple = (12e-3, 12e-3, 3e-3),
                         reconstruction_grid_bounds_mm: Optional[tuple] = None,
                         data_sign=-1,
                         sound_speed_mm_per_s: float = 1525e3,
                         td_focal_length_mm=2.97,
                         recon_mode=4,
                         direct_term_weight=10.0,
                         delay_line_time_s=199 / 3.05e8,
                         preprocess=True,
                         preprocess_bandpass_freq_hz=(15e6, 42e6, 120e6),
                         return_reconstruction_grid=False,
                         verbose=True):
    """ Wrapper around the SAFT algorithm for the .mat file format.

    The data should be stored in the following fields:
        S: Recorded A-lines (n_sensor_positions x n_samples)
        positionXY: xy-positions of the sensor (n_sensor_positions x 2)
        Fs: Sampling frequency of the transducer [Hz]
        trigDelay: Number of samples waited between laser trigger and recording
    """
    if isinstance(signal_path, Path):
        signal_path = str(signal_path)

    raw_signal_dict = h5.loadmat(signal_path)

    assert data_sign in [-1, 1], "data_sign must be either -1 or 1"

    signal = raw_signal_dict['S'] * data_sign
    sensor_positions = raw_signal_dict['positionXY']
    sampling_freq_hz = raw_signal_dict['Fs'].item()
    trigger_delay = raw_signal_dict['trigDelay'].item()

    return saft(signal=signal,
                sensor_positions=sensor_positions,
                sensitivity_field=sensitivity_field,
                reconstruction_grid_spacing_mm=reconstruction_grid_spacing_mm,
                reconstruction_grid_bounds_mm=reconstruction_grid_bounds_mm,
                sampling_freq_hz=sampling_freq_hz,
                td_focal_length_mm=td_focal_length_mm,
                trigger_delay=trigger_delay,
                delay_line_time_s=delay_line_time_s,
                sound_speed_mm_per_s=sound_speed_mm_per_s,
                recon_mode=recon_mode,
                direct_term_weight=direct_term_weight,
                preprocess=preprocess,
                preprocess_bandpass_freq_hz=preprocess_bandpass_freq_hz,
                return_reconstruction_grid=return_reconstruction_grid,
                verbose=verbose)


def saft(signal: ndarray,
         sensor_positions: ndarray,
         sensitivity_field: SensitivityField,
         reconstruction_grid_spacing_mm: tuple = (12e-3, 12e-3, 3e-3),
         reconstruction_grid_bounds_mm: Optional[tuple] = None,
         sampling_freq_hz: float = 1e9,
         td_focal_length_mm: float = 2.97,
         trigger_delay: float = 2080.0,
         delay_line_time_s: float = 199 / 3.05e8,
         sound_speed_mm_per_s: float = 1525e3,
         recon_mode: int = 4,
         direct_term_weight: float = 10.0,
         preprocess: bool = True,
         preprocess_bandpass_freq_hz: tuple = (15e6, 42e6, 120e6),
         return_reconstruction_grid: bool = False,
         verbose: bool = True):
    """
    SAFT (Synthetic Aperture Focusing Technique) / Delay-and-Sum algorithm with sensitivity field weighting.

    [1] M. Schwartz. "Multispectral Optoacoustic Dermoscopy: Methods and Applications"
        (https://mediatum.ub.tum.de/1324031)
    [2] D.M. Soliman. "Augmented microscopy: Development and application of
        high-resolution optoacoustic and multimodal imaging techniques for label-free
        biological observation" (https://mediatum.ub.tum.de/1328957)


    Parameters
    ----------
    signal : ndarray
       Input signal in 2D format (n_sensor_positions x n_samples)
    sensor_positions : ndarray
       xy-positions of sensor measurements (n_sensor_positions x 2)
    sensitivity_field : ndarray
        Sensitivity field object for the used transducer
    reconstruction_grid_spacing_mm: Tuple
        Spacing of the reconstruction grid in x, y and z direction [mm]
    reconstruction_grid_bounds_mm: Tuple
        Boundaries of the reconstruction grid in x, y and z direction [mm]
    sampling_freq_hz : float
        Sampling frequency of the transducer [Hz]
    td_focal_length_mm: float
        Focal length of the transducer [mm]
    trigger_delay: int
        Number of samples waited between laser trigger and recording
    delay_line_time_s: float
        Propagation time of acoustic waves in the glass delay line of the transducer [s]
    sound_speed_mm_per_s: float
        Assumed speeed of sound [mm/s]
    direct_term_weight: float
        Weight of the direct term
    preprocess: bool
        Apply preprocessing (line filter + bandpass filter) to raw signal
    preprocess_bandpass_freq_hz: tuple
        All frequency boundaries (f0, f1, ...). Bands will be created for all neighboring
        pairs, so (f0, f1), (f1, f2), ...
    recon_mode: int
        Mode of the reconstruction algorithm:
        (1) Just direct term (Delay+Sum)
        (2) Just derivative term (filtered backprojection)
        (3) Just derivative term with spatial weighting (filtered backprojection * t)
        (4) Both terms (SAFT)
    return_reconstruction_grid: bool
        Return the reconstruction grid
    verbose: bool
        Print additional information

    Returns
    -------
    ndarray
        Reconstructed image of len(f_bandpass)-1 channels
    """
    if not cp.cuda.is_available():
        raise ImportError("Cuda is not available!")

    signal = cp.asarray(signal, dtype=cp.float32)  # already move signal to GPU for preprocessing
    sensor_pos = cp.asarray(sensor_positions, dtype=cp.float32)
    spacing = cp.asarray(reconstruction_grid_spacing_mm)

    xp = cp.get_array_module(signal)

    # ensure correct shapes
    n_sensor, n_samples = signal.shape
    assert sensor_pos.shape == (n_sensor, 2), "signal and sensor_pos must have the same number of sensors"
    assert recon_mode in [1, 2, 3, 4], "recon_mode must be either 1, 2, 3 or 4"

    # common RSOM preprocessing (line + bandpass filter)
    if preprocess:
        signal = xp.ascontiguousarray(preprocess_signal(signal, bandpass_freq_hz=preprocess_bandpass_freq_hz))

    # calculate time points / space positions of samples
    dt_mm = sound_speed_mm_per_s / sampling_freq_hz  # distance between signal samples [mm]
    t_focus = td_focal_length_mm / sound_speed_mm_per_s + delay_line_time_s  # focal time shift [s]
    t = (xp.arange(n_samples, dtype=signal.dtype) + trigger_delay) * dt_mm  # spatial time vector of the signal [mm]
    t_sp = t - t_focus * sound_speed_mm_per_s  # spatial vector zero'd at the focal point [mm]

    if reconstruction_grid_bounds_mm is None:
        sensor_pos_min = sensor_pos.min(0)
        sensor_pos_max = sensor_pos.max(0)
        sensor_bounds = xp.array([sensor_pos_min, sensor_pos_max]).T
        sensor_bounds = xp.vstack([sensor_bounds, [t_sp[0], t_sp[-1]]])

        # slightly increase the grid size to the next multiple of the voxel size
        sensor_bound_size = (sensor_bounds[:, 1] - sensor_bounds[:, 0]) / spacing
        overhead = (xp.ceil(sensor_bound_size) - sensor_bound_size) / 2
        reconstruction_grid_bounds_mm = sensor_bounds + xp.outer(overhead * spacing, xp.array([-1, 1]))

    else:  # check if provided reconstruction grid is valid
        shape = ((reconstruction_grid_bounds_mm[:, 1] - reconstruction_grid_bounds_mm[:, 0]) / spacing)
        assert xp.all(xp.isclose(shape - shape.round(), 0))

    grid_size = ((reconstruction_grid_bounds_mm[:, 1] - reconstruction_grid_bounds_mm[:, 0]) / spacing).round().astype(
        int)
    grid_size = tuple(grid_size.tolist())
    reconstruction_grid = [
        (reconstruction_grid_bounds_mm[i, 0] + xp.arange(grid_size[i]) * spacing[i]).astype(signal.dtype) for i in
        range(3)
    ]

    # different reconstruction modes
    if recon_mode == 1:
        signal *= direct_term_weight
    if recon_mode in [2, 3, 4]:
        signal_deriv = xp.zeros_like(signal)
        signal_deriv[..., :-1] = (signal[..., 1:] - signal[..., :-1]) * sampling_freq_hz
        if recon_mode == 2:
            signal = -signal_deriv
        elif recon_mode == 3:
            signal = - (signal_deriv * t_sp)
        elif recon_mode == 4:
            signal = signal * direct_term_weight - (signal_deriv * t_sp)

        del signal_deriv
        cp.get_default_memory_pool().free_all_blocks()
        cp.fft.config.get_plan_cache().clear()

    sensitivity_field.simulate(reconstruction_grid[2].get(), x_spacing=0.001, verbose=verbose)
    sfield = cp.array(sensitivity_field.field.astype(np.float32))
    sfield_x = cp.array(sensitivity_field.x.astype(np.float32))
    sfield_z = cp.array(sensitivity_field.z.astype(np.float32))
    sfield_width = cp.array(sensitivity_field.field_width.astype(np.float32))

    output = xp.zeros((len(signal),) + grid_size, dtype=signal.dtype)
    run_saft(
        signal,
        sensor_pos,
        reconstruction_grid[0], reconstruction_grid[1], reconstruction_grid[2],
        sfield, sfield_x, sfield_z, sfield_width,
        t_sp[0].item(), dt_mm,
        output,
        verbose
    )

    if return_reconstruction_grid:
        return output, reconstruction_grid
    return output
