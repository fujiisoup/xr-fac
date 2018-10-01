from collections import OrderedDict
import numpy as np
import xarray as xr

from . import utils


def _read_value(lines, cls):
    vals = lines[0].split('=')
    if len(vals) > 1:
        val = cls(vals[1].strip())
    else:
        val = ''
    return val, lines[1:]


def _read_header(lines):
    header = OrderedDict()
    header['FAC'] = lines[0][4:-1]
    lines = lines[1:]
    header['Endian'], lines = _read_value(lines, int)
    header['TSess'], lines = _read_value(lines, int)
    header['Type'], lines = _read_value(lines, int)
    # header['Verbose'], lines = _read_value(lines, int)
    verbose, lines = _read_value(lines, int)
    key = lines[0].split('\t')[0]
    header['atom'] = key.split(' ')[0]
    header['Z'], lines = _read_value(lines, float)
    header['NBlocks'], lines = _read_value(lines, int)
    return header, lines


def load(filename):
    """ read fac output file, detect filetype automatically and return as
    a xarray object.

    Parameters
    ----------
    filename: path to the file

    Returns
    -------
    obj: xr.Dataset
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    header, lines = _read_header(lines)

    if header['Type'] == 1:
        return _read_en(header, lines)
    if header['Type'] == 2:
        return _read_tr(header, lines)

    raise NotImplementedError(
        'File type {} is not yet implemented.'.format(header['Type']))


def en(filename):
    """ read .en file from fac and return as a xarray object.

    Parameters
    ----------
    filename: path to the file

    Returns
    -------
    obj: xr.Dataset
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    header, lines = _read_header(lines)
    return _read_en(header, lines)


def read_tr(filename):
    """ read .tr file from fac and return as a xarray object.

    Parameters
    ----------
    filename: path to the file

    Returns
    -------
    obj: xr.Dataset
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # header
    header, lines = _read_header(lines)
    return _read_tr(header, lines)


def _read_en(header, lines):
    """ private function to read .en file """
    lncomplex, lsname, lname = utils.get_lengths(header['FAC'])

    def read_blocks(lines):
        block = OrderedDict()
        block['nele'], lines = _read_value(lines, int)
        nlev, lines = _read_value(lines, int)
        # convert to array
        block = {key: np.full(nlev, val) for key, val in block.items()}

        lines = lines[1:]  # skip header
        # read the values
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
        for i, line in enumerate(lines):
            if line.strip() == '':  # if empty
                blocks = read_blocks(lines[i+1:])
                return (block, ) + blocks

            block['ilev'][i] = int(line[:7])
            block['ibase'][i] = int(line[8:14])
            block['energy'][i] = float(line[15:30])
            block['parity'][i] = 1 - np.int8(line[30:32]) * 2
            vnl = int(line[33:38])
            block['n'][i] = np.int8(vnl // 100)
            block['l'][i] = np.int8(vnl - block['n'][i] * 100)
            block['j'][i] = np.int8(line[39:42])
            names = line[43:].split('  ')
            names = [name.strip() for name in names if name.strip() != '']
            block['ncomplex'][i] = names[0]
            block['sname'][i] = names[1]
            block['name'][i] = names[2]

        return (block, )

    idx, eng = lines[0].split('=')[1].split(',')
    header['idx_ground'] = int(idx)
    header['eng_ground'] = float(eng)
    lines = lines[2:]

    blocks = read_blocks(lines)
    keys = blocks[0].keys()
    ds = xr.Dataset(
        {k: ('ilev', np.concatenate([bl[k] for bl in blocks]))
         for k in keys}, attrs=header)
    ds = ds.set_coords(['ilev'])
    ds['energy'].attrs['unit'] = 'eV'
    return ds


def _read_tr(header, lines):
    def read_blocks(lines):
        block = OrderedDict()
        block['nele'], lines = _read_value(lines, int)
        ntrans, lines = _read_value(lines, int)
        block['multipole'], lines = _read_value(lines, int)
        block['gauge'], lines = _read_value(lines, int)
        block['mode'], lines = _read_value(lines, int)
        # convert to array
        block = {key: np.full(ntrans, val) for key, val in block.items()}

        # read the values
        block['lower'] = np.zeros(ntrans, dtype=int)
        block['upper'] = np.zeros(ntrans, dtype=int)
        block['strength'] = np.zeros(ntrans, dtype=float)
        block['A'] = np.zeros(ntrans, dtype=float)

        for i, line in enumerate(lines):
            if line.strip() == '':  # if empty
                blocks = read_blocks(lines[i+1:])
                return (block, ) + blocks
            block['upper'][i] = int(line[:7])
            block['lower'][i] = int(line[11:17])
            block['strength'][i] = float(line[34:48])
            block['A'][i] = float(line[48:62])

        return (block, )

    lines = lines[1:]
    blocks = read_blocks(lines)

    keys = blocks[0].keys()
    ds = xr.Dataset(
        {k: ('itrans', np.concatenate([bl[k] for bl in blocks]))
         for k in keys}, attrs=header)
    ds['lower'].attrs['about'] = 'The lower level index of the transition.'
    ds['upper'].attrs['about'] = 'The upper level index of the transition.'
    ds['strength'].attrs['about'] = 'The weighted oscillator strength gf.'
    return ds
