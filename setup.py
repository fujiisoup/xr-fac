#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup
import re

# load version form _version.py
VERSIONFILE = "xrfac/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# module

setup(name='xrfac',
      version=verstr,
      author="Keisuke Fujii",
      author_email="fujii@me.kyoto-u.ac.jp",
      description=("Python library to handle FAC output file with xarray"),
      license="BSD 3-clause",
      keywords="atomic data, machine learning",
      url="http://github.com/fujiisoup/xrfac",
      include_package_data=True,
      ext_modules=[],
      packages=["xrfac", ],
      package_dir={'xrfac': 'xrfac'},
      py_modules=['xrfac.__init__'],
      test_suite='tests',
      install_requires="""
        numpy>=1.11
        xarray>=0.10
        """,
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Physics']
      )