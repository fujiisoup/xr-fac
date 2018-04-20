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


def hartree2eV(hartree):
    """ Convert hartlee to eV """
    return hartree * 27.21138602


