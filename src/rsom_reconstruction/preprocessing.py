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
    np.ndarray
        filtered signal
    """
    inplace = out is not None
    use_cp = isinstance(signal, cp.ndarray)
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
        out[:] = xp.real(xp.fft.ifft2(fftimage))
    else:
        out = xp.real(xp.fft.ifft2(fftimage))

    # if cupy is used, free the unused memory
    if use_cp:
        del fftimage, filter, f_x
        cp.get_default_memory_pool().free_all_blocks()
        cp.fft.config.get_plan_cache().clear()

    if inplace:
        return
    return out


def bandpass_filter(signal: ndarray,
                    order: float = 4.0,
                    f_bandpass: tuple = (15e6, 42e6, 120e6),
                    fs: float = 1e9,
                    apodization: bool = True,
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
    f_bandpass : tuple
       All frequency boundaries (f0, f1, ...). Bands will be created for all neighboring
       pairs, so (f0, f1), (f1, f2), ...
    fs : float
       Sampling frequency the signal was recorded with.
    apodization : bool
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
        len(f_bandpass)-1 filtered signals
    """
    inplace = out is not None
    use_cp = isinstance(signal, cp.ndarray)
    xp = cp.get_array_module(signal)

    if inplace:
        assert isinstance(signal, type(out))
    else:
        out = xp.zeros((len(f_bandpass) - 1,) + signal.shape, dtype=signal.dtype)

    fft = xp.fft

    if apodization:
        apo_filter = xp.arange(signal.shape[-1]) + 1
        apo_filter = xp.exp(-(apo_filter - apodization_fraction * signal.shape[-1]) / apodization_slope)
        apo_filter = 1 - (1 / (1 + apo_filter))
        signal *= apo_filter
        del apo_filter

    f = fft.fftshift(fft.fftfreq(signal.shape[-1], d=1 / fs))
    signal_fft = fft.fftshift(fft.fft(signal, axis=-1))

    for i in range(len(f_bandpass) - 1):
        filter_bp = xp.exp(-(f / f_bandpass[i + 1]) ** order) * (1 - xp.exp(-(f / f_bandpass[i]) ** order))
        out[i] = fft.ifft(fft.ifftshift(signal_fft * filter_bp), axis=-1).real

    # if cupy is used, free the unused memory
    if use_cp:
        del f, filter_bp, signal_fft
        cp.get_default_memory_pool().free_all_blocks()
        cp.fft.config.get_plan_cache().clear()

    if inplace:
        return

    return out


def preprocess_signal(signal, f_bandpass):
    """
    Apply RSOM preprocessing (line filter, bandpass filter) to the input signal.

    Parameters
    ----------
    signal : ndarray
       Input signal in 2D format (n_sensor x n_samples)
    f_bandpass : tuple
       All frequency boundaries (f0, f1, ...). Bands will be created for all neighboring
       pairs, so (f0, f1), (f1, f2), ...

    Returns
    -------
    ndarray
        len(f_bandpass)-1 line+bandpass filtered signals

    """
    use_cp = isinstance(signal, cp.ndarray)
    xp = cp.get_array_module(signal)


    out_linefiltered = xp.zeros_like(signal)
    line_filter(signal, out=out_linefiltered)

    out_bandpassfiltered = xp.empty((len(f_bandpass) - 1,) + signal.shape, dtype=signal.dtype)
    bandpass_filter(out_linefiltered, f_bandpass=f_bandpass, out=out_bandpassfiltered)

    if use_cp:
        del out_linefiltered
        cp.get_default_memory_pool().free_all_blocks()

    return out_bandpassfiltered
