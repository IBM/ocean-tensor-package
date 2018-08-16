import pyOcean_cpu as ocean
import pyOceanNumpy

A = ocean.tensor([2000])
B = A.convertTo('numpy')
C = A.convertTo('numpy')

print(A.storage.refcount-1)

