import pyOcean_cpu as ocean
import numpy as np
import pyOceanNumpy

t = np.asarray([[1,2],[3,4],[5,6]],np.int32)
print(t)

print("====== ocean.asTensor(t) ======")
a = ocean.asTensor(t);
print(a)

print("====== ocean.asTensor([t,a]) ======")
b = ocean.asTensor([t,a])
print(b)

