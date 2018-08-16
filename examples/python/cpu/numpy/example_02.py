## Conversion from Numpy to Ocean with strides
import pyOcean_cpu as ocean
import ocean_numpy
import numpy as np

ocean.setDisplayWidth(60)

A = np.asarray(range(100))
A = A.reshape([10,10])
print(A)

T = ocean.asTensor(A[1:10:3, 1:10:2])
print(T)
print(T.storage)

print(ocean.asTensor(A[1:10:3, 9:0:-2]))
print(ocean.asTensor(A[7:0:-3, 9:0:-2]))
print(ocean.asTensor(A[7:0:-3, 1:10:2]))

