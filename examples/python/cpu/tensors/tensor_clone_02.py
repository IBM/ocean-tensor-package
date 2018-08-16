## Tensor clone with zero strides
import pyOcean_cpu as ocean

a = ocean.tensor([3,4],[1,0], ocean.float)
a.copy([1,2,3])

print(a)
print(a.storage)

b = a.clone()
print(b)
print(b.storage)

c = a.replicate()
print(c)
print(c.storage)
