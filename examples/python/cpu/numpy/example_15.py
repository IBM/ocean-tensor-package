import pyOcean_cpu as ocean
import numpy as np
import pyOceanNumpy

a = np.arange(24).reshape([3,2,4])
print(a)
b = ocean.asTensor(a).reverseAxes2()
print(b)

b.fill(3)
b.sync()

print(a)

