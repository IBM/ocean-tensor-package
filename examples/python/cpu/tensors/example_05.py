## Tensor with irregular size
import pyOcean_cpu as ocean

t = ocean.tensor([2,3],[12, 3], 1, ocean.int16);
s = t.storage
s.dtype = ocean.int16
s.asTensor().fill(3)
print(t)
print(s)
print(s.size)
print(s.nelem)

