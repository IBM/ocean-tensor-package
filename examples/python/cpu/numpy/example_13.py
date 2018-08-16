import pyOcean_cpu as ocean
import numpy as np
import pyOceanNumpy

a = ocean.tensor([3])
a.fill(2)

b = np.asarray([[1,2],[3,4]],np.int32)

c = ocean.asTensor(a); print(c)
d = ocean.asTensor(b); print(d)
