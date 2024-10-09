import cupy as cp
import numpy as np
from .preprocessing import line_filter, bandpass_filter, preprocess_signal
from time import time
from .sensitivity import SensitivityField
from .utils import ndarray
from typing import Optional, Union

import matplotlib.pyplot as plt
from .saft_cuda import run_saft
from math import sqrt, copysign, floor
import hdf5storage as h5

def saft_munich_adapter(signal_path: str,
                        sensitivity_field: SensitivityField,
                        reconstruction_grid_bounds: Optional[tuple] = None,
                        reconstruction_grid_spacing: tuple = (12e-3, 12e-3, 3e-3),
                        data_sign=-1,
                        sound_speed_mm_per_s: float = 1525e3,
                        focal_length_mm=2.97,
                        detector_diameter_mm=3.0,
                        direct_term_weight=10.0,
                        t_dl=6.524590163934427e-07,
                        preprocess=True,
                        preproces_f_bandpass=(15e6, 42e6, 120e6), verbose=True):
    raw_signal_dict = h5.loadmat(signal_path)

    assert data_sign in [-1, 1], "data_sign must be either -1 or 1"

    signal = raw_signal_dict['S'] * data_sign
    sensor_pos = raw_signal_dict['positionXY']
    f_s = raw_signal_dict['Fs'].item()
    trigger_delay = raw_signal_dict['trigDelay'].item()

    return saft(signal=signal,
                sensitivity_field=sensitivity_field,
                sensor_pos=sensor_pos,
                reconstruction_grid_bounds=reconstruction_grid_bounds,
                reconstruction_grid_spacing=reconstruction_grid_spacing,
                f_s=f_s,
                focal_length_mm=focal_length_mm,
                trigger_delay=trigger_delay,
                t_dl=t_dl,
                sound_speed_mm_per_s=sound_speed_mm_per_s,
                direct_term_weight=direct_term_weight,
                preprocess=preprocess,
                preprocess_f_bandpass=preproces_f_bandpass, verbose=verbose)


# def run_saft_with_sensitivity_py(signal, sensor_pos, voxel_pos_x, voxel_pos_y, voxel_pos_z, sfield, sfield_x, sfield_z, sfield_width, t_0, dt,
#                                  output):
#     n_channels, n_sensor, n_samples = signal.shape
#     n_x, n_y, n_z = voxel_pos_x.size, voxel_pos_y.size, voxel_pos_z.size
#     n_x_sens = sfield.shape[1]
#
#     signal = signal.reshape(-1)
#     sensor_pos = sensor_pos.reshape(-1)
#     sfield = sfield.reshape(-1)
#     output = output.reshape(-1)
#
#     for x in range(n_x):
#         for y in range(n_y):
#             for z in range(n_z):
#                 channelsize = n_x * n_y * n_z
#                 vox_index = z + n_z * y + n_z * n_y * x
#
#                 position_x = voxel_pos_x[x]
#                 position_y = voxel_pos_y[y]
#                 position_z = voxel_pos_z[z]
#
#                 sensitivity_field_width = sfield_width[z]
#
#                 all_sensor_sum = cp.zeros(3)
#                 weight_sum = 0.0
#
#                 for s in range(n_sensor):
#                     dx = position_x - sensor_pos[2 * s]
#                     dy = position_y - sensor_pos[2 * s + 1]
#
#                     r = dx * dx + dy * dy
#                     if r > sensitivity_field_width * sensitivity_field_width:
#                         continue
#
#                     delay = sqrt(dx * dx + dy * dy + position_z * position_z)
#                     delay = (cp.copysign(delay, position_z) - t_0) / dt
#
#                     if (delay < 0 or delay >= n_samples):
#                         continue
#
#                     r = cp.sqrt(r) * 1000
#
#                     delay_floor = cp.floor(delay)
#                     delay_int = delay.astype(int)
#                     r_floor = cp.floor(r)
#                     r_int = int(r)
#
#                     lower_sensitivity_value = sfield[r_int + z * n_x_sens]
#                     upper_sensitivity_value = sfield[r_int + 1 + z * n_x_sens]
#                     sensitivity_value = lower_sensitivity_value * (r_floor + 1 - r) + upper_sensitivity_value * (r - r_floor)
#
#                     for c in range(n_channels):
#                         lower_value = signal[c * n_samples * n_sensor + delay_int + s * n_samples]
#                         upper_value = signal[c * n_samples * n_sensor + delay_int + 1 + s * n_samples]
#                         value = lower_value * (delay_floor + 1 - delay) + upper_value * (delay - delay_floor)
#
#                         all_sensor_sum[c] += value * sensitivity_value
#                         weight_sum += sensitivity_value
#
#
#                 for c in range(n_channels):
#                     if weight_sum > 0.0:
#                         output[vox_index + channelsize * c] = all_sensor_sum[c] / weight_sum

def saft(signal: ndarray,
         sensor_pos: ndarray,
         reconstruction_grid_bounds: tuple,
         reconstruction_grid_spacing: tuple = (12e-3, 12e-3, 3e-3),
         sensitivity_field: Optional[SensitivityField] = None,
         f_s: float = 1e9,
         focal_length_mm: float = 2.97,
         trigger_delay: float = 2080.0,
         t_dl: float = 6.524590163934427e-07,

         sound_speed_mm_per_s: float = 1525e3,
         direct_term_weight=10.0,

         preprocess=True,
         preprocess_f_bandpass=(15e6, 42e6, 120e6),
         recon_mode=4,
         verbose=True):
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
       Input signal in 2D format (n_sensor x n_samples)
    sensor_pos : ndarray
       xy-positions of sensor measurements (n_sensor x 2)
    reconstruction_grid_bounds: Tuple
        ...
    reconstruction_grid_spacing: Tuple
        ...
    sensitivity_field : np.ndarray
        ...
    fs : float
        Sampling frequency [Hz]
    focal_length_mm: float
        Focal length of the transducer [mm]
    detector_diameter_mm: float
        Diameter of the detector [mm]
    trigger_delay: int
        Number of samples waited between laser trigger and recording
    t_dl: float
        Propagation time of acoustic waves in the glass delay line of the transducer
    x_lim_mm: np.ndarray
        Boundaries of the sensor grid in x-direction [mm]
    y_lim_mm: np.ndarray
        Boundaries of the sensor grid in y-direction [mm]
    dx_mm: float
        Spacing of the measurements in x-direction [mm]
    dy_mm: float
        Spacing of the measurements in y-direction [mm]

    sound_speed_mm_per_s: float
        Assumed speeed of sound [mm/s]
    direct_term_weight: float
        Weight of the direct term
    lateral_spacing_mm: float
        Lateral (xy) spacing of voxels in the reconstruction grid [mm]
    axial_spacing_mm: float
        Axial (z) spacing of voxels in the reconstruction grid [mm]

    preprocess: bool
        Apply preprocessing (line filter + bandpass filter) to raw signal
    f_bandpass: tuple
        All frequency boundaries (f0, f1, ...). Bands will be created for all neighboring
        pairs, so (f0, f1), (f1, f2), ...
    recon_mode: int
        Mode of the reconstruction algorithm:
        (1) Just direct term (Delay+Sum)
        (2) Just derivative term (filtered backprojection)
        (3) Just derivative term with spatial weighting (filtered backprojection * t)
        (4) Both terms (SAFT)
    gpu: bool
        Use the GPU

    Returns
    -------
    np.ndarray
        Reconstructed image of len(f_bandpass)-1 channels
    """
    use_cp = True  # TODO: implement CPU version

    if use_cp and not cp.cuda.is_available():
        raise ImportError("Cuda is not available!")

    if use_cp and not isinstance(signal, cp.ndarray):
        signal = cp.asarray(signal, dtype=cp.float32)  # already move signal to GPU for preprocessing
        sensor_pos = cp.asarray(sensor_pos, dtype=cp.float32)
    elif isinstance(signal, np.ndarray):
        signal = signal.astype(np.float32) # signal should be floating type
        sensor_pos = sensor_pos.astype(np.float32)

    xp = cp.get_array_module(signal)

    # ensure correct shapes
    n_sensor, n_samples = signal.shape
    assert sensor_pos.shape[0] == n_sensor, "signal and sensor_pos must have the same number of sensors"
    assert recon_mode in [1, 2, 3, 4], "recon_mode must be either 1, 2, 3 or 4"

    # common RSOM preprocessing (line + bandpass filter)
    if preprocess:
        signal = xp.ascontiguousarray(preprocess_signal(signal, f_bandpass=preprocess_f_bandpass))

    # calculate time points / space positions of samples
    dt_mm = sound_speed_mm_per_s / f_s  # distance between signal samples [mm]
    t_focus = focal_length_mm / sound_speed_mm_per_s + t_dl  # focal time shift [s]
    t = (xp.arange(n_samples, dtype=signal.dtype) + trigger_delay) * dt_mm  # spatial time vector of the signal [mm]
    t_sp = t - t_focus * sound_speed_mm_per_s  # spatial vector zero'd at the focal point [mm]

    spacing = xp.asarray(reconstruction_grid_spacing)
    if reconstruction_grid_bounds is None:
        sensor_pos_min = sensor_pos.min(0)
        sensor_pos_max = sensor_pos.max(0)
        sensor_bounds = xp.array([sensor_pos_min, sensor_pos_max]).T
        sensor_bounds = xp.vstack([sensor_bounds, [t_sp[0], t_sp[-1]]])

        sensor_bound_size = (sensor_bounds[:, 1]-sensor_bounds[:, 0]) / spacing
        overhead = (cp.ceil(sensor_bound_size) - sensor_bound_size) / 2
        reconstruction_grid_bounds = sensor_bounds + xp.outer(overhead * spacing, xp.array([-1, 1]))

    else:  # check if provided reconstruction grid is valid
        shape = ((reconstruction_grid_bounds[:, 1] - reconstruction_grid_bounds[:, 0]) / spacing)
        assert xp.all(xp.isclose(shape - shape.round(), 0))

    grid_size = ((reconstruction_grid_bounds[:, 1] - reconstruction_grid_bounds[:, 0]) / spacing).round().astype(int)
    grid_size = tuple(grid_size.tolist())

    reconstruction_grid_shape = ((reconstruction_grid_bounds[:, 1] - reconstruction_grid_bounds[:, 0]) / spacing).round().astype(int)
    reconstruction_grid = [
        (reconstruction_grid_bounds[i, 0] + xp.arange(grid_size[i]) * spacing[i]).astype(signal.dtype) for i in range(3)
    ]

    # different reconstruction modes
    if recon_mode == 1:
        signal *= direct_term_weight
    if recon_mode in [2, 3, 4]:
        signal_deriv = xp.zeros_like(signal)
        signal_deriv[..., :-1] = (signal[..., 1:] - signal[..., :-1]) * f_s
        if recon_mode == 2:
            signal = -signal_deriv
        elif recon_mode == 3:
            signal = - (signal_deriv * t_sp)
        elif recon_mode == 4:
            signal = signal * direct_term_weight - (signal_deriv * t_sp)

        del signal_deriv
        if use_cp:
            cp.get_default_memory_pool().free_all_blocks()

    output = xp.zeros((len(signal),) + grid_size, dtype=signal.dtype)

    sensitivity_field.simulate(reconstruction_grid[2].get(), x_spacing=0.001, verbose=verbose)

    sfield = sensitivity_field.field.astype(np.float32)
    sfield_x = sensitivity_field.x.astype(np.float32)
    sfield_z = sensitivity_field.z.astype(np.float32)
    sfield_width = sensitivity_field.field_width.astype(np.float32)
    if use_cp:
        sfield = cp.array(sfield)
        sfield_x = cp.array(sfield_x)
        sfield_z = cp.array(sfield_z)
        sfield_width = cp.array(sfield_width)

    run_saft(
        signal,
        sensor_pos,
        reconstruction_grid[0], reconstruction_grid[1], reconstruction_grid[2],
        sfield, sfield_x, sfield_z, sfield_width,
        t_sp[0].item(), dt_mm,
        output,
        verbose
        )

    output = output.get()
    plt.imshow(output[0].max(1))
    plt.show()
    plt.imshow(output[1].max(1))
    plt.show()
    return output
