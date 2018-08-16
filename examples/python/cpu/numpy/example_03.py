## Conversion from Numpy to Ocean, byte-swapped half-precision data
import pyOcean_cpu as ocean
import ocean_numpy
import numpy as np

ocean.setDisplayWidth(60)

values = [0, 1.2, -2.3e4, 7.2e-5, 3.25e-7, np.nan, np.inf, -np.inf]

# Create the arrays
dt = np.dtype(np.float16).newbyteorder('S')
A = np.asarray(values, dtype=np.float16);
B = np.asarray(values, dtype=dt)

# Wrap as Ocean tensors
print(ocean.asTensor(A))
print(ocean.asTensor(B))
