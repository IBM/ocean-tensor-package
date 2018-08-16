## Square-root operation with overlap
import pyOcean_cpu as ocean

a = ocean.asTensor([4,9,16,25])
a1 = ocean.tensor(a.storage,0,[3])
a2 = ocean.tensor(a.storage,1,[3])
print(a1)
print(a2)

ocean.sqrt(a1,a2)
print(a1)
print(a2)

