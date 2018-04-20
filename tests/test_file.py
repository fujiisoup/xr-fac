import os
import numpy as np
import pytest
import xrfac


THIS_DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize('files', [
    ('ne.lev', 'ne.lev.b'),
    ('ne.tr', 'ne.tr.b')])
def test(files):
    ascii_file = THIS_DIR + '/example_data/' + files[0]
    binary_file = THIS_DIR + '/example_data/' + files[1]

    ds_from_ascii = xrfac.ascii.load(ascii_file)
    ds_from_binary = xrfac.binary.load(binary_file)
    for k in ds_from_ascii.variables:
        if ds_from_ascii[k].dtype.kind in 'iuf':
            print(ds_from_ascii[k])
            print(ds_from_binary[k])
            assert np.allclose(ds_from_ascii[k], ds_from_binary[k])
        else:
            assert (ds_from_ascii[k] == ds_from_binary[k]).all()
