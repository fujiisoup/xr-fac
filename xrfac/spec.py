"""
Spectrum related functions
"""
import numpy as np


def convolve_spectrum(x, y, x_dest, dx, resolution=10):
    """
    Convolve line emissions and construct a spectrum.
    
    Parameters
    ----------
    x: 1d-array
        Position of the emission, e.g. energy or wavelength
    y: 1d-array
        Emission strength.
    x_dest: 1d-np.ndarray
        Destination of the spectrum. This must be evenly spaced and 
        monotonically increasing.
    dx: float
        Wavelength resolution of the spectrum
    resolution: int, default 10.
        How many virtual bins are used per bin in x_dest.
        
    Returns
    -------
    convolved: np.ndarray 
        Convolved spectrum, the same size to x_dest.
    """
    diff = np.mean(np.diff(x_dest))
    assert diff > 0.0        
    assert np.allclose(np.diff(x_dest), diff)  # should be evenly spaced

    hist_range = (x_dest[0] - 0.5*diff, x_dest[-1] + 0.5*diff)
    bins = len(x_dest) * resolution
    
    hist, _ = np.histogram(x, bins=bins, range=hist_range, weights=y)
    n_width = dx / diff * resolution
    #  consider +/- 3-sigma range
    x_inst = np.linspace(-3.0*dx, 3.0*dx, int(n_width * 6))  
    gaussian = np.exp(-0.5*(x_inst / dx)**2) / (np.sqrt(2.0 * np.pi) * dx)
    return np.convolve(hist, gaussian, mode='same')[resolution//2::resolution]
    