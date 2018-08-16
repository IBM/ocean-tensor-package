# Numpy interface

The `pyOceanNumpy` module provides conversions between Numpy arrays and Ocean tensors, which are registered in the core library. The functions can also be imported using the `ocean_numpy` module, which simply includes the `pyOceanNumpy` library.

```python
>>> import ocean_numpy
>>> import ocean
>>> ocean.getImportTypes()
['numpy']
>>> ocean.getExportTypes()
['numpy']
```

As a first example we create a shallow copy of a Numpy array using `asTensor`

```python
>>> import numpy as np
>>> import ocean
>>> import ocean_numpy
>>> 
>>> A = np.asarray([[1,2,3],[4,5,6]],dtype=np.float)
>>> T = ocean.asTensor(A)
>>> T
(:,:)
   1   2   3
   4   5   6
<tensor.double of size 2x3 on cpu>
>>> T.storage
   1   2   3   4   5   6
<storage.double of size 6 on cpu>
>>> T.storage.owner 
False
>>> 
>>> A[:] = 3
>>> T
(:,:)
   3   3   3
   3   3   3
<tensor.double of size 2x3 on cpu>
>>> 
>>> T.zero()
>>> A
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])
```

Once the module is loaded, Numpy objects can be used in most places where Ocean or Python objects are accepted:

```python
>>> import numpy as np
>>> import ocean_numpy
>>> import ocean
>>> a = np.arange(6,dtype=np.int8)
>>> b = np.empty(6,dtype=np.float32)
>>> ocean.sqrt(a,b)
>>> print(b)
[ 0.          1.          1.41421354  1.73205078  2.          2.23606801]
>>>
>>> print(ocean.asTensor([b,(range(5) + [np.int8(5)]),ocean.ones(6)],'r'))
(:,:)
   0.00000   1.00000   1.41421   1.73205   2.00000   2.23607
   0.00000   1.00000   2.00000   3.00000   4.00000   5.00000
   1.00000   1.00000   1.00000   1.00000   1.00000   1.00000
<tensor.double of size 3x6 on cpu>
```

The conversion routines can properly deal with positive and negative strides:

```python
>>> import numpy as np
>>> import ocean_numpy
>>> import ocean
>>> 
>>> ocean.setDisplayWidth(60)  # Set the maximum output width
>>> 
>>> A = np.asarray(range(100))
>>> A = A.reshape([10,10])
>>> print(A)
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
>>> 
>>> T = ocean.asTensor(A[1:10:3, 1:10:2])
>>> print(T)
(:,:)
   11   13   15   17   19
   41   43   45   47   49
   71   73   75   77   79
<tensor.int64 of size 3x5 on cpu>
>>> print(T.storage)
   11   12   13   14   15   16   17   18   19   20   21   22
   23   24   25   26   27   28   29   30   31   32   33   34
   35   36   37   38   39   40   41   42   43   44   45   46
   47   48   49   50   51   52   53   54   55   56   57   58
   59   60   61   62   63   64   65   66   67   68   69   70
   71   72   73   74   75   76   77   78   79
<storage.int64 of size 69 on cpu>
>>> 
>>> ocean.asTensor(A[1:10:3, 9:0:-2])
(:,:)
   19   17   15   13   11
   49   47   45   43   41
   79   77   75   73   71
<tensor.int64 of size 3x5 on cpu>
```

Byte-swapped data is also supported:

```python
>>> import numpy as np
>>> import ocean_numpy
>>>
>>> values = [0, 1.2, -2.3e4, 7.2e-5, 3.25e-7, np.nan, np.inf, -np.inf]
>>>
>>> # Create the arrays
>>> dt = np.dtype(np.float16).newbyteorder('S')
>>> A = np.asarray(values, dtype=np.float16);
>>> B = np.asarray(values, dtype=dt)
>>>
>>> # Wrap as Ocean tensors
>>> print(ocean.asTensor(A))
    0.00000e+0   1.20020e+0  -2.30080e+4   7.20024e-5
    2.98023e-7          nan          inf         -inf
<tensor.half of size 8 on cpu>
>>> print(ocean.asTensor(B))
    0.00000e+0   1.20020e+0  -2.30080e+4   7.20024e-5
    2.98023e-7          nan          inf         -inf
<tensor.half of size 8 on cpu (byteswapped)>
```

Numpy scalars are imported as Ocean scalars:

```python
>>> print(ocean.chalf(np.float32(5)))
5 + 0j
>>> a = ocean.tensor([np.int8(2),np.int16(3)])
>>> a.size
(2, 3)
```

Conversion back to Numpy is also provided by the module:

```python
>>> import numpy as np
>>> import ocean_numpy
>>> import ocean
>>> 
>>> A = ocean.arange(24).reshape([4,6])
>>> B = A[1:3,1:5].convertTo("numpy")
>>> print(A)
(:,:)
    0    4    8   12   16   20
    1    5    9   13   17   21
    2    6   10   14   18   22
    3    7   11   15   19   23
<tensor.int64 of size 4x6 on cpu>
>>> print(B)
[[ 5  9 13 17]
 [ 6 10 14 18]]
>>> B[:] = 0
>>> print(A)
(:,:)
    0    4    8   12   16   20
    1    0    0    0    0   21
    2    0    0    0    0   22
    3    7   11   15   19   23
<tensor.int64 of size 4x6 on cpu>
```
