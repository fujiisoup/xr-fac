ATOMIC_SYMBOLS = [
    '__dummy__',  # no Z=0
    # period 1
    'H', 'He',
    # period 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # period 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # period 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
    'Ge', 'As', 'Se', 'Br', 'Kr',
    # period 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # period 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Ti', 'Pb', 'Bi', 'Po',
    'At', 'Rn'
]


C = 2.99792458e8  # light speed in m/s
H = 6.62607004e-34  # planck's constant m2 kg/s
KB = 1.38064852e-23  # boltzmann's constant in m2 kg s^(-2) K^-1
J2EV = 6.24150962915265e18  # eV / J
ALPHA = 7.2973525664e-3  # fine structure constant


def hartree2eV(hartree):
    """ Convert hartlee to eV """
    return hartree * 27.21138602


def eV2hartree(eV):
    """ Convert hartlee to eV """
    return eV / 27.21138602


def eV2nm(eV):
    """ Convert eV to nm """
    # h c / lambda
    # eV -> J : 0.1602e-18
    hc = H * C * 1.0e9 * J2EV  # to nm
    return hc / eV
