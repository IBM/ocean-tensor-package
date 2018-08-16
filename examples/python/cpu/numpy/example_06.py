## Conversion from Ocean to Numpy
import pyOcean_cpu as ocean
import ocean_numpy
import numpy as np

A = ocean.asTensor([[1,2,3],[4,5,6]], ocean.float)
B = A.convertTo('numpy', True)

A.fill(3)
print(B)
print(B.dtype)

A = ocean.asTensor([[1,2,3],[4,5,6]], ocean.float)
A.byteswap()
B = A.convertTo('numpy', True)

print(B)

