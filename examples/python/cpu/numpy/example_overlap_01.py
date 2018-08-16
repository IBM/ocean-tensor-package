import pyOcean_cpu as ocean
import pyOceanNumpy
import numpy as np

a = np.asarray(range(24)).reshape(4,6)
print(a)

b = a[0:3,:]
c = a[1:4,:]
print(b)
print(c)

s = ocean.asTensor(b)
t = ocean.asTensor(c)

print(s)
print(t)

t.storage.copy(s.storage)
print(t)


