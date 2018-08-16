## Cloning of tensors
import pyOcean_cpu as ocean


a = ocean.asTensor([[1,2,3],[4,5,6]], "R", ocean.float)

print(a)
print(a.clone())
print(a.clone(ocean.cpu))

