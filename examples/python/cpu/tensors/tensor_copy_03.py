import pyOcean_cpu as ocean

a = ocean.asTensor([1,2,3])
b = ocean.tensor(a.size, a.dtype)

a.byteswap()
b.copy(a)

print(a)
print(b)
