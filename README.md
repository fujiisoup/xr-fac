# xr-fac

A small library to read output files of [the flexible atomic code (FAC)](https://github.com/flexible-atomic-code/fac).

This library especially utilizes [xarray](http://xarray.pydata.org/), which can handle labeled ND-array.

# Install

```
python setup.py install
```

# Usage

The usage is simple,
```python
>>> import xrfac
>>> xrfac.binary.load('tests/example_data/ne.lev.b')
<xarray.Dataset>
Dimensions:   (ilev: 37)
Coordinates:
  * ilev      (ilev) int64 0 1 2 3 4 5 6 7 8 9 ... 27 28 29 30 31 32 33 34 35 36
Data variables:
    nele      (ilev) int64 10 10 10 10 10 10 10 10 ... 10 10 10 10 10 10 10 10
    ibase     (ilev) int64 -1 -1 -1 -1 -1 -1 -1 -1 ... -1 -1 -1 -1 -1 -1 -1 -1
    energy    (ilev) float64 0.0 724.3 726.4 737.0 ... 938.7 938.8 939.1 943.9
    parity    (ilev) int8 1 -1 -1 -1 -1 1 1 1 1 1 1 ... 1 1 -1 -1 -1 -1 1 1 1 1
    n         (ilev) int8 2 3 3 3 3 3 3 3 3 3 3 3 3 ... 3 3 3 3 3 3 3 3 3 3 3 3
    l         (ilev) int8 1 0 0 0 0 1 1 1 1 1 1 1 1 ... 2 2 0 0 1 1 1 1 2 2 2 2
    j         (ilev) int8 0 4 2 0 2 2 4 6 2 4 0 2 2 ... 6 2 2 0 0 2 4 2 2 4 6 4
    ncomplex  (ilev) <U32 '1*2 2*8' '1*2 2*7 3*1' ... '1*2 2*7 3*1'
    sname     (ilev) <U24 '2p6' '2p5 3s1' '2p5 3s1' ... '2s1 3d1' '2s1 3d1'
    name      (ilev) <U56 '2p+4(0)0' '2p+3(3)3 3s+1(1)4' ... '2s+1(1)1 3d+1(5)4'
Attributes:
    TSess:       1524228903
    FAC:         1.1.4
    Type:        1
    Z:           26.0
    atom:        Fe
    Endian:      True
    NBlocks:     1
    idx_ground:  0
    eng_ground:  -31229.537665885982
>>>
```
`xrfac.binary.load` function loads a fac output file (binary) into the memory
and convert to an xarray Dataset object.

If the file is very large, we provides out-of-memory function, where file that
does not fit into the memory can be loaded lazily,

```python
>>> xrfac.binary.load('tests/example_data/ne.lev.b', in_memory=False)
<xarray.Dataset>
Dimensions:   (ilev: 37)
Coordinates:
  * ilev      (ilev) int32 0 1 2 3 4 5 6 7 8 9 ... 27 28 29 30 31 32 33 34 35 36
Data variables:
    name      (ilev) object dask.array<shape=(37,), chunksize=(37,)>
    ncomplex  (ilev) object dask.array<shape=(37,), chunksize=(37,)>
    sname     (ilev) object dask.array<shape=(37,), chunksize=(37,)>
    nele      (ilev) int32 dask.array<shape=(37,), chunksize=(37,)>
    ibase     (ilev) int32 dask.array<shape=(37,), chunksize=(37,)>
    energy    (ilev) float64 dask.array<shape=(37,), chunksize=(37,)>
    parity    (ilev) int8 dask.array<shape=(37,), chunksize=(37,)>
    n         (ilev) int8 dask.array<shape=(37,), chunksize=(37,)>
    l         (ilev) int8 dask.array<shape=(37,), chunksize=(37,)>
    j         (ilev) int8 dask.array<shape=(37,), chunksize=(37,)>
Attributes:
    TSess:       1524228903
    FAC:         1.1.4
    Type:        1
    Z:           26.0
    atom:        Fe
    Endian:      1
    NBlocks:     1
    idx_ground:  0
    eng_ground:  -31229.537665885982
>>>
```
Note that [dask](http://dask.pydata.org/) and
[netcdf4](http://unidata.github.io/netcdf4-python/) need to be installed
for out-of-memory usage.
