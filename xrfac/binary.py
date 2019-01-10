from collections import OrderedDict
import tempfile
import struct
import numpy as np
import xarray as xr
from . import utils


ONE_FILE_ENTRIES = 1000


def _F_header(file):
    """ Read common header from file """
    header = OrderedDict()
    header['TSess'] = struct.unpack('l', file.read(8))[0]
    major_ver = struct.unpack('i', file.read(4))[0]
    minor_ver = struct.unpack('i', file.read(4))[0]
    micro_ver = struct.unpack('i', file.read(4))[0]
    header['FAC'] = '{}.{}.{}'.format(major_ver, minor_ver, micro_ver)
    header['Type'] = struct.unpack('i', file.read(4))[0]

    header['Z'] = struct.unpack('f', file.read(4))[0]
    header['atom'] = file.read(2).decode('utf-8')
    file.read(1)
    header['Endian'] = bool(file.read(1))
    header['NBlocks'] = struct.unpack('i', file.read(4))[0]
    return header, file


def load(filename, in_memory=True):
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
        if header['Type'] == 7:
            return _read_sp(header, f, in_memory=in_memory)

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


def tr(filename, in_memory):
    """ read .tr file from fac and return as a xarray object.

    Parameters
    ----------
    filename: path to the file

    Returns
    -------
    obj: xr.DataArray
    """
    if not has_xarray:
        raise ImportError('This function requires xarray installed in the '
                          'environment.')

    with open(filename, 'rb') as f:
        header, f = _F_header(f)
        return _read_tr(header, f, in_memory)


def _read_en(header, file, in_memory):
    lncomplex, lsname, lname = utils.get_lengths(header['FAC'])

    def read_block(file):
        block = OrderedDict()
        position = struct.unpack('l', file.read(8))[0]
        length = struct.unpack('l', file.read(8))[0]
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
            xr.concat(datasets, dim='ilev').to_netcdf(
                utils._NeverCloseFile(outfile))
            files.append(outfile)

        filenames = [f.name for f in files]
        ds = xr.open_mfdataset(filenames)
        ds.attrs._temporary_files = filenames  # for testing

    ionization_eng = ds['energy'].min()
    ds.attrs['idx_ground'] = ds['energy'].argmin().values.item()
    ds.attrs['eng_ground'] = ionization_eng.values.item()
    ds['energy'] -= ionization_eng
    ds['energy'].attrs['unit'] = 'eV'
    return ds


def _read_tr(header, file, in_memory):

    def read_block(file):
        block = OrderedDict()
        position = struct.unpack('l', file.read(8))[0]
        length = struct.unpack('l', file.read(8))[0]
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
        while i < header['NBlocks']:
            count = 0
            datasets = []
            while count < ONE_FILE_ENTRIES and i < header['NBlocks']:
                ds = to_xarray(read_block(file))
                count += len(ds['itrans'])
                i += 1
                datasets.append(ds)
            outfile = tempfile.NamedTemporaryFile()
            xr.concat(datasets, dim='itrans').to_netcdf(
                utils._NeverCloseFile(outfile))
            files.append(outfile)

        filenames = [f.name for f in files]
        ds = xr.open_mfdataset(filenames)
        ds.attrs._temporary_files = filenames  # for testing
        return ds


def _read_sp(header, file, in_memory):
    lncomplex, lsname, lname = utils.get_lengths(header['FAC'])

    def read_block(file):
        block = OrderedDict()
        position = struct.unpack('l', file.read(8))[0]
        length = struct.unpack('l', file.read(8))[0]
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
        return xr.concat(
            [to_xarray(read_block(file)) for i in range(header['NBlocks'])],
            dim='itrans')
    else:
        tempfile.mkdtemp()
        files = []
        i = 0
        while i < header['NBlocks']:
            count = 0
            datasets = []
            while count < ONE_FILE_ENTRIES and i < header['NBlocks']:
                ds = to_xarray(read_block(file))
                count += len(ds['itrans'])
                i += 1
                datasets.append(ds)
            outfile = tempfile.NamedTemporaryFile()
            xr.concat(datasets, dim='itrans').to_netcdf(
                utils._NeverCloseFile(outfile))
            files.append(outfile)

        filenames = [f.name for f in files]
        ds = xr.open_mfdataset(filenames)
        ds.attrs._temporary_files = filenames  # for testing
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
