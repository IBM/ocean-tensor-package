import pyOcean_cpu as ocean

a = ocean.asTensor([1,2,3])
s = a.storage
b = ocean.int8(9)

x = ocean.cast(a, a.dtype, ocean.cpu)
print(x.obj == a.obj)
print(x.storage.obj == a.storage.obj)
print(x)

x = ocean.cast(s, s.dtype, ocean.cpu)
print(x is s)
print(x.obj == s.obj)
print(x)

x = ocean.cast(b, b.dtype)
print(x)

print(ocean.cast(1, ocean.int8))
print(ocean.cast([1,2,3], ocean.float))

