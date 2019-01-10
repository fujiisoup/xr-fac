import os
import warnings
import numpy as np
import pytest
import xrfac


THIS_DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize('files', [
     ('ne.lev', 'ne.lev.b'),
     ('ne.tr', 'ne.tr.b'),
     ('Ne03a.en', 'Ne03b.en'),  # 1.1.5
     ('resulta.sp', 'resultb.sp'),
     ])
def test(files):
    ascii_file = THIS_DIR + '/example_data/' + files[0]
    binary_file = THIS_DIR + '/example_data/' + files[1]

    ds_from_ascii = xrfac.ascii.load(ascii_file)
    ds_from_binary = xrfac.binary.load(binary_file)
    for k in ds_from_binary.variables:
        if ds_from_ascii[k].dtype.kind in 'iuf':
            if k in ['strength', 'rrate', 'trate']:
                assert np.allclose(ds_from_ascii[k], ds_from_binary[k],
                                   rtol=1e-4)
            else:
                assert np.allclose(ds_from_ascii[k], ds_from_binary[k])
        else:
            assert (ds_from_ascii[k] == ds_from_binary[k]).all()

    ds_oufofmemory = xrfac.binary.load(binary_file, in_memory=False)
    for k in ds_from_binary.variables:
        if ds_oufofmemory[k].dtype.kind in 'iuf':
            assert np.allclose(ds_oufofmemory[k], ds_from_binary[k])
        else:
            assert (ds_oufofmemory[k] == ds_from_binary[k]).all()

    # can be load
    ds_oufofmemory.load()
    # make sure the temporary files should not be there
    for f in ds_oufofmemory.attrs._temporary_files:
        assert not os.path.exists(f)
    # can be save as another netcdf
    ds_oufofmemory.to_netcdf('tmp.nc')
    os.remove('tmp.nc')


def test_tr():
    tr_ascii_file = THIS_DIR + '/example_data/ne_multipole.tr'
    tr_bin_file = THIS_DIR + '/example_data/ne_multipole.tr.b'
    en_bin_file = THIS_DIR + '/example_data/ne.lev.b'

    ds_ascii = xrfac.ascii.load(tr_ascii_file)
    ds_bin = xrfac.binary.load(tr_bin_file)
    ds_bin_en = xrfac.binary.load(en_bin_file)

    ds_bin = xrfac.binary.oscillator_strength(ds_bin, ds_bin_en)
    for k in ds_bin.variables:
        if ds_ascii[k].dtype.kind in 'iuf':
            assert np.allclose(ds_ascii[k], ds_bin[k])
        else:
            assert (ds_ascii[k] == ds_bin[k]).all()


@pytest.mark.parametrize('files', [
    ('ne.tr', 'ne.tr.b', 'ne.lev.b'),
    ('ne_multipole.tr', 'ne_multipole.tr.b', 'ne.lev.b'),
    ])
def test_tr_A(files):
    tr_ascii_file = THIS_DIR + '/example_data/' + files[0]
    tr_bin_file = THIS_DIR + '/example_data/' + files[1]
    en_bin_file = THIS_DIR + '/example_data/' + files[2]

    tr_ascii = xrfac.ascii.load(tr_ascii_file)
    tr_bin = xrfac.binary.load(tr_bin_file)
    en_bin = xrfac.binary.load(en_bin_file)

    A = xrfac.utils.getA(en_bin, tr_bin)
    assert np.allclose(A, tr_ascii['A'])
