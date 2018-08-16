## Conversion from storage to tensor
import pyOcean_cpu as ocean

s = ocean.storage(14);
s.dtype = ocean.cfloat
s.asTensor().fill(1+2j)
print(s)

s.dtype = ocean.float
print(s)

t = ocean.tensor(s, 0, [3,4])
print(t)

t = ocean.tensor(s, 4, [3,4], 'F', 1)
print(t)

t = ocean.tensor(s, 4, [3,2], 'F', 1, ocean.cfloat)
print(t)

