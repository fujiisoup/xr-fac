import numpy as np
from xrfac import utils


def test_atomic_symbols():
    assert utils.ATOMIC_SYMBOLS.index('Ar') == 18
    assert utils.ATOMIC_SYMBOLS.index('Kr') == 36
    assert utils.ATOMIC_SYMBOLS.index('Pd') == 46
    assert utils.ATOMIC_SYMBOLS.index('Lu') == 71
    assert utils.ATOMIC_SYMBOLS.index('W') == 74
    assert utils.ATOMIC_SYMBOLS.index('Rn') == 86
