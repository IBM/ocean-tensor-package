import pyOcean_cpu as ocean

# Fortran-style strides
a = ocean.tensor([3,4],'F');
a.copy(range(a.nelem))
print(a)
print(a.strides)

b = a.reshape([2,3,2])
print(b)
print(b.storage.obj == a.storage.obj)

a.reshape([2,6],True)
print(a)


# C-style strides
a = ocean.tensor([3,4],'C');
a.copy(range(a.nelem))
print(a)
print(a.strides)

b = a.reshape([2,3,2])
print(b)
print(b.storage.obj == a.storage.obj)

a.reshape([2,6],True)
print(a)

