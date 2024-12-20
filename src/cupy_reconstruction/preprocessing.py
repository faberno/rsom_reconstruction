from typing import Union, List, Optional
import numpy as np
import cupy as cp
import gc
from .utils import ndarray

def line_filter(signal: ndarray,
                sigma: float = 75.0,
                out: Optional[ndarray] = None) -> Optional[ndarray]:
    """
    Reflection-Line Filter: Suppresses frequencies responsible for line artifacts
    in the 2D B-Scan

    Parameters
    ----------
    signal : ndarray
       Input signal in 2D format (n_sensor x n_samples)
    sigma : float
       Filter width (larger = stronger filter)
    out : ndarray
        Output array the final result is written into (optional)

    Returns
    -------
    ndarray
        filtered signal
    """
    inplace = out is not None
    gpu = isinstance(signal, cp.ndarray)
    xp = cp.get_array_module(signal)

    if inplace:
        assert isinstance(signal, type(out))

    N, M = signal.shape

    f_x = xp.arange(M) + 1
    filter = xp.maximum(xp.exp(-f_x ** 2 / (2 * (M / sigma) ** 2)),
                        xp.exp(-(f_x - M) ** 2 / (2 * (M / sigma) ** 2)), dtype=signal.dtype)

    fftimage = xp.fft.fft2(signal)
    fftimage[0] *= filter
    if inplace:
        out[:] = xp.fft.ifft2(fftimage).real
    else:
        out = xp.fft.ifft2(fftimage).real

    # if cupy is used, free the unused memory
    if gpu:
        del fftimage, filter, f_x
        cp.get_default_memory_pool().free_all_blocks()
        cp.fft.config.get_plan_cache().clear()

    return out


def bandpass_filter(signal: ndarray,
                    order: float = 4.0,
                    bandpass_freq_hz: tuple = (15e6, 42e6, 120e6),
                    sampling_freq_hz: float = 1e9,
                    apply_apodization: bool = True,
                    apodization_fraction: float = 0.9,
                    apodization_slope: float = 10.0,
                    out: Optional[ndarray] = None) -> Optional[List[ndarray]]:
    """
    Bandpass Filter: Divides the signal into multiple frequency bands

    Parameters
    ----------
    signal : ndarray
       Input signal in 2D format (n_sensor x n_samples)
    order : float
       Order of exponential filter
    bandpass_freq_hz : tuple
       All frequency boundaries (f0, f1, ...). Bands will be created for all neighboring
       pairs, so (f0, f1), (f1, f2), ...
    sampling_freq_hz : float
       Sampling frequency the signal was recorded with.
    apply_apodization : bool
       Enable sigmoid apodization.
    apodization_fraction : float
       After which fraction of the signal the apodization should start.
    apodization_slope : float
       Strength of the sigmoid slope in the apodization filter.
    out : ndarray
        Output array/tensor the final result is written into (optional)
    Returns
    -------
    ndarray
        len(bandpass_freq_hz)-1 filtered signals
    """
    inplace = out is not None
    gpu = isinstance(signal, cp.ndarray)
    xp = cp.get_array_module(signal)

    if inplace:
        assert isinstance(signal, type(out))
    else:
        out = xp.zeros((len(bandpass_freq_hz) - 1,) + signal.shape, dtype=signal.dtype)

    fft = xp.fft

    if apply_apodization:
        apo_filter = xp.arange(signal.shape[-1]) + 1
        apo_filter = xp.exp(-(apo_filter - apodization_fraction * signal.shape[-1]) / apodization_slope)
        apo_filter = 1 - (1 / (1 + apo_filter))
        signal *= apo_filter
        del apo_filter

    f = fft.fftshift(fft.fftfreq(signal.shape[-1], d=1 / sampling_freq_hz))
    signal_fft = fft.fftshift(fft.fft(signal, axis=-1))

    for i in range(len(bandpass_freq_hz) - 1):
        filter_bp = xp.exp(-(f / bandpass_freq_hz[i + 1]) ** order) * (1 - xp.exp(-(f / bandpass_freq_hz[i]) ** order))
        out[i] = fft.ifft(fft.ifftshift(signal_fft * filter_bp), axis=-1).real

    # if cupy is used, free the unused memory
    if gpu:
        del f, filter_bp, signal_fft
        cp.get_default_memory_pool().free_all_blocks()
        cp.fft.config.get_plan_cache().clear()

    return out


def preprocess_signal(signal, **kwargs):
    """
    Apply RSOM preprocessing (line filter, bandpass filter) to the input signal.

    Parameters
    ----------
    signal : ndarray
       Input signal in 2D format (n_sensor x n_samples)

    kwargs
        Parameters for the line and bandpass filter:
            - sigma: float
                (Line Filter) Filter width (larger = stronger filter)
            - order: float
                (Bandpass Filter) Order of exponential filter
            - bandpass_freq_hz: tuple
                (Bandpass Filter) All frequency boundaries (f0, f1, ...). Bands will be created for all neighboring
                pairs, so (f0, f1), (f1, f2), ...
            - sampling_freq_hz: float
                (Bandpass Filter) Sampling frequency the signal was recorded with.
            - apodization: bool
                (Bandpass Filter) Enable sigmoid apodization.
            - apodization_fraction: float
                (Bandpass Filter) After which fraction of the signal the apodization should start.
            - apodization_slope: float
                (Bandpass Filter) Strength of the sigmoid slope in the apodization filter.

    Returns
    -------
    ndarray
        len(f_bandpass)-1 line+bandpass filtered signals

    """
    gpu = isinstance(signal, cp.ndarray)

    # apply line filter
    line_filter_parameter = ['sigma']
    line_filter_kwargs = {k: v for k, v in kwargs.items() if k in line_filter_parameter}
    signal_linefiltered = line_filter(signal, **line_filter_kwargs)

    # apply bandpass filter
    bandpass_filter_parameter = ['order', 'f_bandpass', 'fs', 'apodization', 'apodization_fraction', 'apodization_slope']
    bandpass_filter_kwargs = {k: v for k, v in kwargs.items() if k in bandpass_filter_parameter}
    signal_bandpassfiltered = bandpass_filter(signal_linefiltered, **bandpass_filter_kwargs)

    if gpu:
        del signal_linefiltered
        cp.get_default_memory_pool().free_all_blocks()
        cp.fft.config.get_plan_cache().clear()

    return signal_bandpassfiltered
