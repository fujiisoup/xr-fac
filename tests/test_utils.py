import numpy as np
from xrfac import utils


def test_atomic_symbols():
    assert utils.ATOMIC_SYMBOLS.index('Ar') == 18
    assert utils.ATOMIC_SYMBOLS.index('Kr') == 36
    assert utils.ATOMIC_SYMBOLS.index('Pd') == 46
    assert utils.ATOMIC_SYMBOLS.index('Lu') == 71
    assert utils.ATOMIC_SYMBOLS.index('W') == 74
    assert utils.ATOMIC_SYMBOLS.index('Rn') == 86


def test_atomic_data():
    # carbon
    data = utils.ATOMIC_DATA[6]
    assert data[1]['protons'] == 6
    assert data[1]['nucleons'] == 13
    assert data[1]['is_radioactive'] == False
    assert data[1]['spin_quantum_number'] == 0.5
    assert np.allclose(data[1]['nuclear_g_factor'], 1.4048236)
    assert np.allclose(data[1]['natural_abundance'], 0.0107)
    assert np.allclose(data[1]['elecric_quadrupole_moment'], 0.0)
