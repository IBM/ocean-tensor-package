import pyOcean_cpu as ocean

a = ocean.tensor([3,4])
a.copy(range(a.nelem))

b = a.asContiguous()
print(a.obj == b.obj)
print(b)

b = a.asContiguous('C')
print(a.obj == b.obj)
print(b)

b = a.asContiguous('F')
print(a.obj == b.obj)
print(b.strides)
print(b)

