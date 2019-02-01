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


@pytest.mark.parametrize('file', ['Ne03b.ham', 'Al.ham'])
def test_ham(file):
    binary_file = THIS_DIR + '/example_data/' + file

    ds_from_binary = xrfac.binary.load_ham(binary_file)
    ds_oufofmemory = xrfac.binary.load_ham(binary_file, in_memory=False)
    for k in ds_from_binary.coords:
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
    # at least have some non-diagonal entries
    nondiag= ds_oufofmemory.isel(
        entry=ds_oufofmemory['i'] != ds_oufofmemory['j'])
    assert len(nondiag) > 0

    # set_index -> unstacking should work
    unstacked = ds_oufofmemory.set_index(
        entry=['i', 'j', 'sym_index']).unstack('entry')


@pytest.mark.parametrize('file', ['Ne03b.ham', 'Al.ham'])
def test_ham_basis(file):
    binary_file = THIS_DIR + '/example_data/' + file

    _, ds_from_binary = xrfac.binary.load_ham(binary_file, return_basis=True)
    _, ds_oufofmemory = xrfac.binary.load_ham(
        binary_file, return_basis=True, in_memory=False)
    for k in ds_from_binary.coords:
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


@pytest.mark.parametrize('files', [
    ('O.basis', 'O.ham')
])
def test_basis(files):
    basis_file = THIS_DIR + '/example_data/' + files[0]
    ham_file = THIS_DIR + '/example_data/' + files[1]
    basis = xrfac.ascii.load_basis(basis_file)
    ham, basis_list = xrfac.binary.load_ham(ham_file, return_basis=True)
    print(ham)

    # sym_index in hamiltonian should be included in basis
    assert ham['sym_index'].isin(basis['sym_index']).all()
    for sym in np.unique(ham['sym_index']):
        # maximum index in hamiltonian should not exceed number of basis
        ham1 = ham.isel(entry=ham['sym_index'] == sym)
        bas1 = basis.isel(ibasis=basis['sym_index'] == sym)
        assert (bas1['i'] <= ham1['i'].max()).all()
        assert (bas1['i'] <= ham1['j'].max()).all()
