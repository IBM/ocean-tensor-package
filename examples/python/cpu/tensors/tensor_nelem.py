## tensor.nelem - number of elemens in a tensor
import pyOcean_cpu as ocean

a = ocean.tensor([2,3,4])
b = ocean.tensor([2,0])
c = ocean.tensor([])

print(a.nelem)
print(b.nelem)
print(c.nelem)

