from collections import OrderedDict
import tempfile
import struct
import pathlib
import numpy as np
import xarray as xr
from . import utils


ONE_FILE_ENTRIES = 1000
MAX_SYMMETRIES = 256

LONGTYPE = 'l' if struct.calcsize('l') == 8 else 'll'
MAXLEVEB = 1000000


def _F_header(file):
    """ Read common header from file """
    header = OrderedDict()
    header['TSess'] = struct.unpack(LONGTYPE, file.read(8))[0]
    major_ver = struct.unpack('i', file.read(4))[0]
    minor_ver = struct.unpack('i', file.read(4))[0]
    micro_ver = struct.unpack('i', file.read(4))[0]
    header['FAC'] = '{}.{}.{}'.format(major_ver, minor_ver, micro_ver)
    header['Type'] = struct.unpack('i', file.read(4))[0]

    header['Z'] = struct.unpack('f', file.read(4))[0]
    header['atom'] = file.read(2).decode('utf-8')
    file.read(1)
    header['Endian'] = 'True' if bool(file.read(1)) else 'False'
    header['NBlocks'] = struct.unpack('i', file.read(4))[0]
    return header, file


def load(filename, in_memory=True, **kwargs):
    """ read fac output file, detect filetype automatically and return as
    a xarray object.

    Parameters
    ----------
    filename: path to the file
    in_memory: boolean
        If True, load the file into memory. If False, the file is once
        converted to netCDF format and saved to temporal location.
        Then, the lazy load will be performed.
        Note that for in_memory=False, dask needs to be installed.
    
    Other Parameters
    ----------------
    only_pop: 
        If True, returns only population, valid for .sp file

    Returns
    -------
    obj: xr.DataArray
    """
    with open(filename, 'rb') as f:
        header, f = _F_header(f)
        if header['Type'] == 1:
            return _read_en(header, f, in_memory=in_memory)
        if header['Type'] == 2:
            return _read_tr(header, f, in_memory=in_memory)
        if header['Type'] == 5:
            return _read_ai(header, f, in_memory=in_memory)
        if header['Type'] == 7:
            return _read_sp(header, f, in_memory=in_memory, **kwargs)
        if header['Type'] == 12:
            return _read_enEB(header, f, in_memory=in_memory, **kwargs)
        if header['Type'] == 13:
            return _read_trEB(header, f, in_memory=in_memory, **kwargs)

        raise NotImplementedError(
            'File type {} is not yet implemented.'.format(header['Type']))


def en(filename, in_memory):
    """ read .en file from fac and return as a xarray object.

    Parameters
    ----------
    filename: path to the file

    Returns
    -------
    obj: xr.DataArray
    """
    with open(filename, 'rb') as f:
        header, f = _F_header(f)
        return _read_en(header, f, in_memory)


def enEB(filename, in_memory):
    """ read .en file from fac (saved with StructureEB) and return as a xarray object.

    Parameters
    ----------
    filename: path to the file

    Returns
    -------
    obj: xr.DataArray
    """
    with open(filename, 'rb') as f:
        header, f = _F_header(f)
        return _read_enEB(header, f, in_memory)


def tr(filename, in_memory):
    """ read .tr file from fac and return as a xarray object.

    Parameters
    ----------
    filename: path to the file

    Returns
    -------
    obj: xr.DataArray
    """
    with open(filename, 'rb') as f:
        header, f = _F_header(f)
        return _read_tr(header, f, in_memory)


def trEB(filename, in_memory):
    """ read .tr file with Field (saved with TRTableEB) from fac and return as a xarray object.

    Parameters
    ----------
    filename: path to the file

    Returns
    -------
    obj: xr.DataArray
    """
    with open(filename, 'rb') as f:
        header, f = _F_header(f)
        return _read_trEB(header, f, in_memory)


def _read_en(header, file, in_memory):
    lncomplex, lsname, lname = utils.get_lengths(header['FAC'])

    def read_block(file):
        block = OrderedDict()
        position = struct.unpack(LONGTYPE, file.read(8))[0]
        length = struct.unpack(LONGTYPE,file.read(8))[0]
        block['nele'] = struct.unpack('i', file.read(4))[0]
        nlev = struct.unpack('i', file.read(4))[0]

        # convert to array
        block = {key: np.full(nlev, val) for key, val in block.items()}

        block['ilev'] = np.zeros(nlev, dtype=int)
        block['ibase'] = np.zeros(nlev, dtype=int)
        block['energy'] = np.zeros(nlev, dtype=float)
        block['parity'] = np.zeros(nlev, dtype=np.int8)
        block['n'] = np.zeros(nlev, dtype=np.int8)
        block['l'] = np.zeros(nlev, dtype=np.int8)
        block['j'] = np.zeros(nlev, dtype=np.int8)
        block['ncomplex'] = np.chararray(nlev, itemsize=lncomplex,
                                         unicode=True)
        block['sname'] = np.chararray(nlev, itemsize=lsname, unicode=True)
        block['name'] = np.chararray(nlev, itemsize=lname, unicode=True)

        for i in range(nlev):
            p = struct.unpack('h', file.read(2))[0]
            parity = np.sign(p)
            p = p * parity
            n = np.int8(p // 100)
            l = np.int8(p - n * 100)
            parity = 0 if parity > 0 else 1
            block['parity'][i], block['n'][i], block['l'][i] = parity, n, l
            block['j'][i] = struct.unpack('h', file.read(2))[0]
            block['ilev'][i] = struct.unpack('i', file.read(4))[0]
            block['ibase'][i] = struct.unpack('i', file.read(4))[0]
            block['energy'][i] = struct.unpack('d', file.read(8))[0]
            block['ncomplex'][i] = file.read(lncomplex).strip(b'\x00').strip()
            block['sname'][i] = file.read(lsname).strip(b'\x00').strip()
            block['name'][i] = file.read(lname).strip(b'\x00').strip()

        return block

    def to_xarray(block):
        keys = block.keys()
        ds = xr.Dataset(
            {k: ('ilev', block[k]) for k in keys}, attrs=header)
        ds = ds.set_coords(['ilev'])
        ds['energy'] = utils.hartree2eV(ds['energy'])
        return ds

    if in_memory:
        ds = xr.concat(
            [to_xarray(read_block(file)) for i in range(header['NBlocks'])],
            dim='ilev')
    else:
        tempdir = tempfile.TemporaryDirectory()
        files = []
        i = 0
        while i < header['NBlocks']:
            count = 0
            datasets = []
            while count < ONE_FILE_ENTRIES and i < header['NBlocks']:
                ds = to_xarray(read_block(file))
                count += len(ds['ilev'])
                i += 1
                datasets.append(ds)
            outfile = tempdir.name + '{}.nc'.format(i)
            xr.concat(datasets, dim='ilev').to_netcdf(outfile)
            files.append(outfile)

        ds = xr.open_mfdataset(files)
        ds.attrs['_temporary_files'] = files  # for testing

    ionization_eng = ds['energy'].min()
    ds.attrs['idx_ground'] = ds['energy'].argmin('ilev').values.item()
    ds.attrs['eng_ground'] = ionization_eng.values.item()
    ds['energy'] -= ionization_eng
    ds['energy'].attrs['unit'] = 'eV'
    return ds


def _read_enEB(header, file, in_memory):
    lncomplex, lsname, lname = utils.get_lengths(header['FAC'])

    def read_block(file):
        block = OrderedDict()
        position = struct.unpack(LONGTYPE, file.read(8))[0]
        length = struct.unpack(LONGTYPE, file.read(8))[0]
        block['nele'] = struct.unpack('i', file.read(4))[0]
        nlev = struct.unpack('i', file.read(4))[0]
        block['efield'] = struct.unpack('d', file.read(8))[0]
        block['bfield'] = struct.unpack('d', file.read(8))[0]
        block['fangle'] = struct.unpack('d', file.read(8))[0]

        # convert to array
        block = {key: np.full(nlev, val) for key, val in block.items()}

        block['ilevEB'] = np.zeros(nlev, dtype=int)
        block['energy'] = np.zeros(nlev, dtype=float)
        block['ilev'] = np.zeros(nlev, dtype=int)
        block['M'] = np.zeros(nlev, dtype=int)

        for i in range(nlev):
            block['ilevEB'][i] = struct.unpack('i', file.read(4))[0]
            block['energy'][i] = struct.unpack('d', file.read(8))[0]
            k = struct.unpack('i', file.read(4))[0]
            block['ilev'][i] = np.abs(k) % MAXLEVEB
            block['M'][i] = (np.abs(k) // MAXLEVEB) * np.sign(k)

        return block

    def to_xarray(block):
        keys = block.keys()
        ds = xr.Dataset(
            {k: ('ilevEB', block[k]) for k in keys}, attrs=header)
        ds = ds.set_coords(['ilevEB'])
        ds['energy'] = utils.hartree2eV(ds['energy'])
        return ds

    if in_memory:
        ds = xr.concat(
            [to_xarray(read_block(file)) for i in range(header['NBlocks'])],
            dim='ilevEB')
    else:
        files = []
        i = 0
        while i < header['NBlocks']:
            count = 0
            datasets = []
            while count < ONE_FILE_ENTRIES and i < header['NBlocks']:
                ds = to_xarray(read_block(file))
                count += len(ds['ilev'])
                i += 1
                datasets.append(ds)
            outfile = tempfile.NamedTemporaryFile()
            xr.concat(datasets, dim='ilevEB').to_netcdf(outfile.name)
            files.append(outfile)

        filenames = [f.name for f in files]
        ds = xr.open_mfdataset(filenames)
        ds.attrs['_temporary_files'] = filenames  # for testing

    ionization_eng = ds['energy'].min()
    ds.attrs['idx_ground'] = ds['energy'].argmin().values.item()
    ds.attrs['eng_ground'] = ionization_eng.values.item()
    ds['energy'] -= ionization_eng
    ds['energy'].attrs['unit'] = 'eV'
    return ds


def _read_tr(header, file, in_memory):

    def read_block(file):
        block = OrderedDict()
        position = struct.unpack(LONGTYPE,file.read(8))[0]
        length = struct.unpack(LONGTYPE,file.read(8))[0]
        block['nele'] = struct.unpack('i', file.read(4))[0]
        ntrans = struct.unpack('i', file.read(4))[0]
        block['gauge'] = struct.unpack('i', file.read(4))[0]
        block['mode'] = struct.unpack('i', file.read(4))[0]
        block['multipole'] = struct.unpack('i', file.read(4))[0]
        is_multipole = block['multipole'] != 0
        # convert to array
        block = {key: np.full(ntrans, val) for key, val in block.items()}

        # read the values
        block['lower'] = np.zeros(ntrans, dtype=int)
        block['upper'] = np.zeros(ntrans, dtype=int)
        strength_key = 'M' if is_multipole else 'strength'
        block[strength_key] = np.zeros(ntrans, dtype=np.float32)

        for i in range(ntrans):
            block['lower'][i] = struct.unpack('i', file.read(4))[0]
            block['upper'][i] = struct.unpack('i', file.read(4))[0]
            block[strength_key][i] = struct.unpack('f', file.read(4))[0]
        return block

    def to_xarray(block):
        keys = block.keys()
        ds = xr.Dataset(
            {k: ('itrans', block[k]) for k in keys}, attrs=header)
        ds['lower'].attrs['about'] = 'The lower level index of the transition.'
        ds['upper'].attrs['about'] = 'The upper level index of the transition.'
        if 'strength' in ds:
            ds['strength'].attrs['about'] = 'The weighted oscillator strength gf.'
        if 'M' in ds:
            ds['M'].attrs['about'] = 'The multipole matrix elements M.'
        return ds

    if in_memory:
        return xr.concat(
            [to_xarray(read_block(file)) for i in range(header['NBlocks'])],
            dim='itrans')
    else:
        files = []
        i = 0
        tempdir = tempfile.TemporaryDirectory()
        while i < header['NBlocks']:
            count = 0
            datasets = []
            while count < ONE_FILE_ENTRIES and i < header['NBlocks']:
                ds = to_xarray(read_block(file))
                count += len(ds['itrans'])
                i += 1
                datasets.append(ds)
            outfile = tempdir.name + '/{}.tr'.format(i)
            xr.concat(datasets, dim='itrans').to_netcdf(outfile)
            files.append(outfile)

        ds = xr.open_mfdataset(files)
        ds.attrs['_temporary_files'] = files  # for testing
        return ds


def _read_ai(header, file, in_memory):

    def read_block(file):
        block = OrderedDict()
        position = struct.unpack(LONGTYPE,file.read(8))[0]
        length = struct.unpack(LONGTYPE,file.read(8))[0]
        block['nele'] = struct.unpack('i', file.read(4))[0]
        ntrans = struct.unpack('i', file.read(4))[0]
        # just emin
        _ = struct.unpack('f', file.read(4))[0]
        negrid = struct.unpack('i', file.read(4))[0]
        # just ignore e_grid
        _ = [struct.unpack('d', file.read(8))[0] for _ in range(negrid)]

        # convert to array
        block = {key: np.full(ntrans, val) for key, val in block.items()}

        # read the values
        block['lower'] = np.zeros(ntrans, dtype=int)
        block['upper'] = np.zeros(ntrans, dtype=int)
        block['rate'] = np.zeros(ntrans, dtype=np.float32)
        for i in range(ntrans):
            block['lower'][i] = struct.unpack('i', file.read(4))[0]
            block['upper'][i] = struct.unpack('i', file.read(4))[0]
            block['rate'][i] = struct.unpack('f', file.read(4))[0]
        return block

    def to_xarray(block):
        keys = block.keys()
        ds = xr.Dataset(
            {k: ('itrans', block[k]) for k in keys}, attrs=header)
        ds['lower'].attrs['about'] = 'The lower (bounded) level index of the transition.'
        ds['upper'].attrs['about'] = 'The upper (free) level index of the transition.'
        ds['rate'] = ds['rate'] * utils.RATE_AU
        ds['rate'].attrs['about'] = 'The autoionization rate.'
        return ds

    if in_memory:
        return xr.concat(
            [to_xarray(read_block(file)) for i in range(header['NBlocks'])],
            dim='itrans')
    else:
        files = []
        i = 0
        tempdir = tempfile.TemporaryDirectory()
        while i < header['NBlocks']:
            count = 0
            datasets = []
            while count < ONE_FILE_ENTRIES and i < header['NBlocks']:
                ds = to_xarray(read_block(file))
                count += len(ds['itrans'])
                i += 1
                datasets.append(ds)
            outfile = tempdir.name + '{}.nc'.format(i)
            xr.concat(datasets, dim='itrans').to_netcdf(outfile)
            files.append(outfile)

        ds = xr.open_mfdataset(files)
        ds.attrs['_temporary_files'] = files  # for testing
        return ds


def _read_sp(header, file, in_memory, only_pop=False):
    lncomplex, lsname, lname = utils.get_lengths(header['FAC'])

    def read_block(file):
        block = OrderedDict()
        position = struct.unpack(LONGTYPE,file.read(8))[0]
        length = struct.unpack(LONGTYPE,file.read(8))[0]
        block['nele'] = struct.unpack('i', file.read(4))[0]
        ntrans = struct.unpack('i', file.read(4))[0]
        block['iblock'] = struct.unpack('i', file.read(4))[0]
        block['fblock'] = struct.unpack('i', file.read(4))[0]
        block['icomplex'] = file.read(lncomplex).strip(b'\x00').strip()
        block['fcomplex'] = file.read(lncomplex).strip(b'\x00').strip()
        block['TYPE'] = struct.unpack('i', file.read(4))[0]

        # convert to array
        block = {key: np.full(ntrans, val) for key, val in block.items()}

        # read the values
        block['lower'] = np.zeros(ntrans, dtype=int)
        block['upper'] = np.zeros(ntrans, dtype=int)
        block['energy'] = np.zeros(ntrans, dtype=float)
        block['strength'] = np.zeros(ntrans, dtype=float)
        block['rrate'] = np.zeros(ntrans, dtype=float)
        block['trate'] = np.zeros(ntrans, dtype=float)

        for i in range(ntrans):
            block['lower'][i] = struct.unpack('i', file.read(4))[0]
            block['upper'][i] = struct.unpack('i', file.read(4))[0]
            block['energy'][i] = struct.unpack('f', file.read(4))[0]
            block['strength'][i] = struct.unpack('f', file.read(4))[0]
            block['rrate'][i] = struct.unpack('f', file.read(4))[0]
            block['trate'][i] = struct.unpack('f', file.read(4))[0]
        return block

    def to_xarray(block):
        keys = block.keys()
        ds = xr.Dataset(
            {k: ('itrans', block[k]) for k in keys}, attrs=header)
        ds['energy'] = utils.hartree2eV(ds['energy'])
        return ds

    if in_memory:
        blocks = []
        for i in range(header['NBlocks']):
            block = read_block(file)
            if only_pop and (block['TYPE'] != 0).all():
                break
            blocks.append(to_xarray(block))
        return xr.concat(blocks, dim='itrans')
    else:
        files = []
        i = 0
        tempdir = tempfile.TemporaryDirectory()
        while i < header['NBlocks']:
            count = 0
            datasets = []
            while count < ONE_FILE_ENTRIES and i < header['NBlocks']:
                block = read_block(file)
                if only_pop and (block['TYPE'] != 0).all():
                    break
                ds = to_xarray(block)
                count += len(ds['itrans'])
                i += 1
                datasets.append(ds)
            outfile = tempdir.name + '/{}.sp'.format(i)
            xr.concat(datasets, dim='itrans').to_netcdf(outfile)
            files.append(outfile)

        ds = xr.open_mfdataset(files, combine='nested', concat_dim='itrans')
        ds.attrs['_temporary_files'] = files  # for testing
        return ds


def oscillator_strength(tr, en):
    """ Add oscillator strength for transition data, based on the equation
    (2.2)
    """
    if 'strength' in tr:
        return tr
    # energy difference
    en = en['energy']
    w = utils.eV2hartree(en.isel(ilev=tr['upper']) - en.isel(ilev=tr['lower']))
    L = np.abs(tr['multipole'])
    strength = 1 / (2.0*L+1) * w * (utils.ALPHA * w)**(2*L-2) * tr['M']**2
    tr['strength'] = strength.astype(np.float32)
    tr['strength'].attrs['about'] = 'The weighted oscillator strength gf.'
    del tr['M']
    return tr


def load_ham(filename, return_basis=False, in_memory=True):
    """ read fac hamiltonian file and return as
    a xarray object.

    Parameters
    ----------
    filename: path to the hamiltonian file
    return_basis: Also returns basis
    in_memory: boolean
        If True, load the file into memory. If False, the file is once
        converted to netCDF format and saved to temporal location.
        Then, the lazy load will be performed.
        Note that for in_memory=False, dask needs to be installed.

    Returns
    -------
    obj: xr.DataArray

    if return_basis is True:
        also returns xr.DataArray
    """
    header = OrderedDict()
    hamiltonian = []
    with open(filename, 'rb') as f:
        header['ng0'] = struct.unpack('i', f.read(4))[0]
        header['ng'] = struct.unpack('i', f.read(4))[0]
        header['kg'] = [struct.unpack('i', f.read(4))[0]
                        for i in range(header['ng'])]
        header['ngp'] = struct.unpack('i', f.read(4))[0]
        header['kgp'] = [struct.unpack('i', f.read(4))[0]
                         for i in range(header['ngp'])]

        return _load_ham(f, header, return_basis, in_memory)


def _load_ham(f, header, return_basis, in_memory):
    """ load hamiltonian and return xarray object """

    def load_sym(sym_index):
        h = OrderedDict()
        s = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        if dim <= 0:
            return None

        orig_dim = struct.unpack('i', f.read(4))[0]
        n_basis = struct.unpack('i', f.read(4))[0]
        basis = [struct.unpack('i', f.read(4))[0] for i in range(n_basis)]
        # number of elements
        n = struct.unpack('i', f.read(4))[0]
        h['i'] = np.zeros(n, int)
        h['j'] = np.zeros(n, int)
        h['value'] = np.zeros(n, float)
        h['sym_index'] = sym_index * np.ones(n, int)
        for i in range(n):
            h['i'][i] = struct.unpack('i', f.read(4))[0]
            h['j'][i] = struct.unpack('i', f.read(4))[0]
            h['value'][i] = struct.unpack('d', f.read(8))[0]
        return h, basis

    def to_xarray(block):
        h, basis = block
        keys = h.keys()
        ds = xr.DataArray(
            h['value'], dims=['entry'], coords={
                k: ('entry', h[k]) for k in keys if k != 'value'},
                attrs=header, name='value')
        basis = xr.DataArray(
            basis, dims=['i'],
            coords={'sym': h['sym_index'][0], 'i': np.arange(len(basis))},
            attrs=header, name='basis')
        return ds, basis

    if in_memory:
        syms = []
        basis = []
        for i in range(MAX_SYMMETRIES):
            sym = load_sym(i)
            if sym is not None:
                h, b = to_xarray(sym)
                syms.append(h)
                basis.append(b)
        if return_basis:
            return xr.concat(syms, dim='entry'), xr.concat(basis, dim='i')
        return xr.concat(syms, dim='entry')
    else:
        files = []
        basis_files = []
        i = 0
        tempdir = tempfile.TemporaryDirectory()
        while i < MAX_SYMMETRIES:
            count = 0
            datasets = []
            basis_sets = []
            while count < ONE_FILE_ENTRIES and i < MAX_SYMMETRIES:
                sym = load_sym(i)
                if sym is not None:
                    ham, basis = to_xarray(sym)
                    datasets.append(ham)
                    basis_sets.append(basis)
                    count += len(ham['entry'])
                i += 1
            outfile = tempdir.name + '/{}.ham'.format(i)
            xr.concat(datasets, dim='entry').to_netcdf(outfile)
            files.append(outfile)
            if return_basis:
                outfile = tempdir.name + '/{}.basis'.format(i)
                xr.concat(basis_sets, dim='i').to_netcdf(outfile)
                basis_files.append(outfile)

        ds = xr.open_mfdataset(files, combine='nested', concat_dim='entry')['value']
        ds.attrs['_temporary_files'] = files  # for testing
        if return_basis:
            basis = xr.open_mfdataset(basis_files, combine='nested', concat_dim='i')['basis']
            basis.attrs['_temporary_files'] = basis_files  # for testing
            return ds, basis

        return ds

def load_basis(filename, return_mixcoef=False, fac_version='1.1.5'):
    """
    read fac basis table file and return as an xarray object.

    Parameters
    ----------
    filename: path to the hamiltonian file
    return_mixcoef: boolean
        if True, also returns mixcoef as xr.Dataset

    Returns
    -------
    obj: xr.Dataset
    """
    # since the file itself is ascii, just call ascii.load_basis
    from . import ascii
    return ascii.load_basis(filename, return_mixcoef, fac_version)
