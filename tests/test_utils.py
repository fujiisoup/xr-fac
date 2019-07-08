import os
import numpy as np
import xrfac
from xrfac import utils

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def test_atomic_symbols():
    assert utils.ATOMIC_SYMBOLS.index('Ar') == 18
    assert utils.ATOMIC_SYMBOLS.index('Kr') == 36
    assert utils.ATOMIC_SYMBOLS.index('Pd') == 46
    assert utils.ATOMIC_SYMBOLS.index('Lu') == 71
    assert utils.ATOMIC_SYMBOLS.index('W') == 74
    assert utils.ATOMIC_SYMBOLS.index('Rn') == 86


def test_nm2eV():
    rng = np.random.RandomState(0)
    for nm in np.exp(rng.randn(100)):
        eV = utils.nm2eV(nm)
        nm2 = utils.eV2nm(eV)
        assert np.allclose(nm, nm2)


def test_decode_pj():
    basis = xrfac.ascii.load_basis(THIS_DIR + '/example_data/O.basis')
    p_expected = basis['p']
    j_expected = basis['j']
    p_actual, j_actual = utils.decode_pj(basis['sym_index'])
    assert np.allclose(p_actual, p_expected)
    assert np.allclose(j_actual, j_expected)
