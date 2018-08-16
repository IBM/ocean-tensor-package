import pyOcean_cpu as ocean

a = ocean.asTensor([1,2,3])
s = a.storage
b = ocean.int8(9)

x = ocean.ensure(a, a.dtype)
print(x.obj == a.obj)
print(x.storage.obj == a.storage.obj)

x = ocean.ensure(a, ocean.float)
print(x)

ocean.ensure(a, ocean.int8, ocean.cpu, True)
print(a)

x = ocean.ensure(s, s.dtype)
print(x is s)
print(x.obj == s.obj)

x = ocean.ensure(s, ocean.half)
print(x)

ocean.ensure(s, ocean.cfloat, True)
print(s)
print(a)

x = ocean.ensure(b, b.dtype)
print(x)

x = ocean.ensure(b, ocean.double)
print(x)

ocean.ensure(b, ocean.float)
print(b)

