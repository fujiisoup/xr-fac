from distutils.version import LooseVersion


def get_lengths(version):
    """ Get version dependent string lengths """
    if LooseVersion(version) < LooseVersion('1.1.5'):
        lncomplex = 32
        lsname = 24
        lname = 56
    else:
        lncomplex = 32
        lsname = 48
        lname = 128
    return lncomplex, lsname, lname


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


ATOMIC_MASS = [
    1.0079,  #	Hydrogen	H	1
    4.0026,  #	Helium	He	2
    6.941,  #	Lithium	Li	3
    9.0122,  #	Beryllium	Be	4
    10.811,  #	Boron	B	5
    12.0107,  #	Carbon	C	6
    14.0067,  #	Nitrogen	N	7
    15.9994,  #	Oxygen	O	8
    18.9984,  #	Fluorine	F	9
    20.1797,  #	Neon	Ne	10
    22.9897,  #	Sodium	Na	11
    24.305,  #	Magnesium	Mg	12
    26.9815,  #	Aluminum	Al	13
    28.0855,  #	Silicon	Si	14
    30.9738,  #	Phosphorus	P	15
    32.065,  #	Sulfur	S	16
    35.453,  #	Chlorine	Cl	17
    39.0983,  #	Potassium	K	19
    39.948,  #	Argon	Ar	18
    40.078,  #	Calcium	Ca	20
    44.9559,  #	Scandium	Sc	21
    47.867,  #	Titanium	Ti	22
    50.9415,  #	Vanadium	V	23
    51.9961,  #	Chromium	Cr	24
    54.938,  #	Manganese	Mn	25
    55.845,  #	Iron	Fe	26
    58.6934,  #	Nickel	Ni	28
    58.9332,  #	Cobalt	Co	27
    63.546,  #	Copper	Cu	29
    65.39,  #	Zinc	Zn	30
    69.723,  #	Gallium	Ga	31
    72.64,  #	Germanium	Ge	32
    74.9216,  #	Arsenic	As	33
    78.96,  #	Selenium	Se	34
    79.904,  #	Bromine	Br	35
    83.8,  #	Krypton	Kr	36
    85.4678,  #	Rubidium	Rb	37
    87.62,  #	Strontium	Sr	38
    88.9059,  #	Yttrium	Y	39
    91.224,  #	Zirconium	Zr	40
    92.9064,  #	Niobium	Nb	41
    95.94,  #	Molybdenum	Mo	42
    98,  #	Technetium	Tc	43
    101.07,  #	Ruthenium	Ru	44
    102.9055,  #	Rhodium	Rh	45
    106.42,  #	Palladium	Pd	46
    107.8682,  #	Silver	Ag	47
    112.411,  #	Cadmium	Cd	48
    114.818,  #	Indium	In	49
    118.71,  #	Tin	Sn	50
    121.76,  #	Antimony	Sb	51
    126.9045,  #	Iodine	I	53
    127.6,  #	Tellurium	Te	52
    131.293,  #	Xenon	Xe	54
    132.9055,  #	Cesium	Cs	55
    137.327,  #	Barium	Ba	56
    138.9055,  #	Lanthanum	La	57
    140.116,  #	Cerium	Ce	58
    140.9077,  #	Praseodymium	Pr	59
    144.24,  #	Neodymium	Nd	60
    145,  #	Promethium	Pm	61
    150.36,  #	Samarium	Sm	62
    151.964,  #	Europium	Eu	63
    157.25,  #	Gadolinium	Gd	64
    158.9253,  #	Terbium	Tb	65
    162.5,  #	Dysprosium	Dy	66
    164.9303,  #	Holmium	Ho	67
    167.259,  #	Erbium	Er	68
    168.9342,  #	Thulium	Tm	69
    173.04,  #	Ytterbium	Yb	70
    174.967,  #	Lutetium	Lu	71
    178.49,  #	Hafnium	Hf	72
    180.9479,  #	Tantalum	Ta	73
    183.84,  #	Tungsten	W	74
    186.207,  #	Rhenium	Re	75
    190.23,  #	Osmium	Os	76
    192.217,  #	Iridium	Ir	77
    195.078,  #	Platinum	Pt	78
    196.9665,  #	Gold	Au	79
    200.59,  #	Mercury	Hg	80
    204.3833,  #	Thallium	Tl	81
    207.2,  #	Lead	Pb	82
    208.9804,  #	Bismuth	Bi	83
    209,  #	Polonium	Po	84
    210,  #	Astatine	At	85
    222,  #	Radon	Rn	86
    223,  #	Francium	Fr	87
    226,  #	Radium	Ra	88
    227,  #	Actinium	Ac	89
    231.0359,  #	Protactinium	Pa	91
    232.0381,  #	Thorium	Th	90
    237,  #	Neptunium	Np	93
    238.0289,  #	Uranium	U	92
    243,  #	Americium	Am	95
    244,  #	Plutonium	Pu	94
    247,  #	Curium	Cm	96
    247,  #	Berkelium	Bk	97
    251,  #	Californium	Cf	98
    252,  #	Einsteinium	Es	99
    257,  #	Fermium	Fm	100
    258,  #	Mendelevium	Md	101
    259,  #	Nobelium	No	102
    261,  #	Rutherfordium	Rf	104
    262,  #	Lawrencium	Lr	103
    262,  #	Dubnium	Db	105
    264,  #	Bohrium	Bh	107
    266,  #	Seaborgium	Sg	106
    268,  #	Meitnerium	Mt	109
    272,  #	Roentgenium	Rg	111
    277,  #	Hassium	Hs	108
    ]
    # Read more: https://www.lenntech.com/periodic/mass/atomic-mass.htm#ixzz5FXLOrRO4


C = 2.99792458e8  # light speed in m/s
H = 6.62607004e-34  # planck's constant m2 kg/s
KB = 1.38064852e-23  # boltzmann's constant in m2 kg s^(-2) K^-1
J2EV = 6.24150962915265e18  # eV / J
ALPHA = 7.2973525664e-3  # fine structure constant
RATE_AU = 4.13413733E16  # inverse of time in atomic unit


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


def getA(levels, transition):
    """ Get Transition rate by from level and transition file """
    L = transition['multipole']
    omega = eV2hartree(levels['energy'][transition['upper']] -
                       levels['energy'][transition['lower']])

    if (transition['multipole'] == 0).all():
        gf = transition['strength']
    elif (transition['multipole'] != 0).all():
        L = transition['multipole']
        gf = (1 / (2 * L + 1) * omega * (ALPHA * omega)**(2 * L - 2) *
              transition['strength']**2)
    else:
        raise ValueError('multipole is expected all zero or all non-zero.')

    gA = 2 * ALPHA**3 * omega**2 * gf * RATE_AU
    g = levels['j'][transition['upper']] + 1
    return gA / g
