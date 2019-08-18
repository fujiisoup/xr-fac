from collections import OrderedDict
import numpy as np
import xarray as xr
import sys

from . import utils


def _read_value(lines, cls):
    vals = lines[0].split('=')
    if len(vals) > 1:
        val = cls(vals[1].strip())
        if cls is str:
            val = val.encode('utf-8')
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
    if header['Type'] == 7:
        return _read_sp(header, lines)

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
            block['parity'][i] = np.int8(line[30:32])
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
                return lines[i+1:], block
            block['upper'][i] = int(line[:7])
            block['lower'][i] = int(line[11:17])
            block['strength'][i] = float(line[34:48])
            block['A'][i] = float(line[48:62])

        return None, block

    lines = lines[1:]
    lines, block = read_blocks(lines)
    blocks = [block]
    while lines is not None:
        lines, block = read_blocks(lines)
        blocks.append(block)

    keys = blocks[0].keys()
    ds = xr.Dataset(
        {k: ('itrans', np.concatenate([bl[k] for bl in blocks]))
         for k in keys}, attrs=header)
    ds['lower'].attrs['about'] = 'The lower level index of the transition.'
    ds['upper'].attrs['about'] = 'The upper level index of the transition.'
    ds['strength'].attrs['about'] = 'The weighted oscillator strength gf.'
    return ds


def _read_sp(header, lines):
    def read_blocks(lines):
        block = OrderedDict()
        block['nele'], lines = _read_value(lines, int)
        ntrans, lines = _read_value(lines, int)
        block['TYPE'], lines = _read_value(lines, int)
        block['iblock'], lines = _read_value(lines, int)
        block['icomplex'], lines = _read_value(lines, str)
        block['fblock'], lines = _read_value(lines, int)
        block['fcomplex'], lines = _read_value(lines, str)
        # convert to array
        block = {key: np.full(ntrans, val) for key, val in block.items()}

        # read the values
        block['upper'] = np.zeros(ntrans, dtype=int)
        block['lower'] = np.zeros(ntrans, dtype=int)
        block['energy'] = np.zeros(ntrans, dtype=float)
        block['strength'] = np.zeros(ntrans, dtype=float)
        block['rrate'] = np.zeros(ntrans, dtype=float)
        block['trate'] = np.zeros(ntrans, dtype=float)

        for i, line in enumerate(lines):
            if line.strip() == '':  # if empty
                return lines[i+1:], block
            block['upper'][i] = int(line[:6])
            block['lower'][i] = int(line[6:13])
            block['energy'][i] = float(line[13:27])
            block['strength'][i] = float(line[27:39])
            block['rrate'][i] = float(line[39:51])
            block['trate'][i] = float(line[51:63])
        return None, block

    lines = lines[1:]
    lines, block = read_blocks(lines)
    blocks = [block]
    while lines is not None:
        lines, block = read_blocks(lines)
        blocks.append(block)

    keys = blocks[0].keys()
    ds = xr.Dataset(
    {k: ('itrans', np.concatenate([bl[k] for bl in blocks]))
     for k in keys}, attrs=header)
    ds['lower'].attrs['about'] = 'The lower level index of the transition.'
    ds['upper'].attrs['about'] = 'The upper level index of the transition.'
    ds['energy'].attrs['about'] = 'The transition energy in eV'
    ds['strength'].attrs['about'] = 'The line luminosity in photon/s'
    return ds


class _BasisLoader(object):
    def __init__(self, fac_version='1.1.5'):
        self.blocks = []
        self.n_states = None
        self.lncomplex, self.lsname, self.lname = utils.get_lengths(
            fac_version)

    def __call__(self, line):
        if line[0] == '#':
            self.n_states = int(line[-6:])
            self.blocks.append(OrderedDict())
            for k in ['sym_index', 'p', 'j', 'k', 'kgroup', 'kcfg', 'kstate',
                      'ncomplex', 'sname', 'name']:
                self.blocks[-1][k] = []
            return
        if line == '\n':
            return
        self.blocks[-1]['sym_index'].append(int(line[:6]))
        self.blocks[-1]['p'].append(int(line[9:11]))
        self.blocks[-1]['j'].append(int(line[12:14]))
        self.blocks[-1]['k'].append(int(line[18:22]))
        self.blocks[-1]['kgroup'].append(int(line[23:26]))
        self.blocks[-1]['kcfg'].append(int(line[27:32]))
        self.blocks[-1]['kstate'].append(int(line[33:38]))
        offset = 41
        self.blocks[-1]['ncomplex'].append(
            line[offset: offset+self.lncomplex].strip())
        offset = offset+self.lncomplex + 1
        self.blocks[-1]['sname'].append(
            line[offset: offset+self.lsname].strip())
        offset = offset+self.lsname + 1
        self.blocks[-1]['name'].append(
            line[offset: offset+self.lname].strip())

    def finalize(self):
        """ Just add an index for each symmetry """
        for block in self.blocks:
            block['i'] = np.arange(len(block['sym_index']))


class _MixcoefLoader():
    pass


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
    if return_mixcoef:
        raise NotImplementedError('return_mixcoef=True is not implemented')

    load_basis = _BasisLoader(fac_version)
    load_mixcoef = _MixcoefLoader()

    loader = None
    with open(filename, 'r') as f:
        for line in f:
            if line[:3] == '===':
                if loader is None:
                    loader = load_basis
                elif loader is load_basis:
                    if not return_mixcoef:
                        break
                    loader = load_mixcoes
            else:
                loader(line)

    load_basis.finalize()
    if not return_mixcoef:
        blocks = load_basis.blocks
        keys = blocks[0].keys()
        ds = xr.Dataset(
            {k: ('ibasis', np.concatenate([bl[k] for bl in blocks]))
             for k in keys})
        return ds


def load_rate(filename):
    """
    read fac rate file and return as an xarray object.

    Parameters
    ----------
    filename: path to rate file.

    Returns
    -------
    obj: xr.Dataset
    """
    upper = []
    lower = []
    rates = []
    inv_rates = []
    nt = 1

    with open(filename) as f:
        lines = f.readlines()

    nt = int(lines[0][33:38].strip())
    temperatures = np.zeros(nt)
    for i in range(nt):
        temperatures[i] = float(lines[i + 1][:12])

    rate = []
    inv_rate = []
    while len(lines) > 0:
        line = lines.pop(0)
        if line[0] == '#':
            lower.append(int(line[1:8]))
            upper.append(int(line[12:17]))
        elif len(line) <= 1:
            if len(rate) > 0:
                assert len(rate) == nt
                assert len(inv_rate) == nt
                rates.append(rate)
                inv_rates.append(inv_rate)
                rate = []
                inv_rate = []
        else:  # rate
            rate.append(float(line[13:24]))
            inv_rate.append(float(line[25:36]))

    if len(rate) > 0:
        assert len(rate) == nt
        assert len(inv_rate) == nt
        rates.append(rate)
        inv_rates.append(inv_rate)

    return xr.Dataset({'rate': (('itrans', 'temperature'), rates),
                       'inv_rate': (('itrans', 'temperature'), inv_rates)},
                      coords={'upper': ('itrans', upper),
                              'lower': ('itrans', lower),
                              'temperature': temperatures})
