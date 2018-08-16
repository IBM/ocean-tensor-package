## Conversion from Numpy arrays to Ocean tensors
import pyOcean_cpu as ocean
import ocean_numpy
import numpy as np

A = np.asarray([[1,2,3],[4,5,6]],dtype=np.float)
T = ocean.asTensor(A, False)

print(T)
print(T.storage)
print(T.storage.owner)

A[:] = 3
print(T)

T.storage.zero()
T.sync()
print(A)

print("\n-------------------------------\n")

A = np.asarray([[1,2,3],[4,5,6]],dtype=np.float)
T = ocean.asTensor(A, True)

print(T)
print(T.storage)
print(T.storage.owner)

A[:] = 3
print(T)

T.storage.zero()
T.sync()
print(A)

