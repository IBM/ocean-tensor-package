## Indexing with a zero-dimensional Boolean array
import pyOcean_cpu as ocean

a = ocean.asTensor(True,ocean.bool);
b = ocean.zeros([3,3])
b[a] = 3
print(b)
