import numpy as np
from xrfac import spec


def test_spec():
    x = np.array([0.0])
    y = np.array([1.0])
    x_dest = np.linspace(-6, 6, 100)
    convolve = spec.convolve_spectrum(x, y, x_dest, dx=0.5, resolution=10)
    assert np.allclose(np.trapz(convolve, x_dest), 1.0, atol=0.01)
    
    x = np.array([0.0, 1.0])
    y = np.array([1.0, 1.5])
    x_dest = np.linspace(-8, 8, 200)
    convolve = spec.convolve_spectrum(x, y, x_dest, dx=0.2, resolution=10)
    assert np.allclose(np.trapz(convolve, x_dest), 2.5, atol=0.1)