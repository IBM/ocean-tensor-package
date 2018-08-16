import pyOcean_cpu as ocean

a = ocean.arange(10,ocean.double)
a.byteswap()
print(a)
print(ocean.sqrt(a))
ocean.sqrt(a,a)
print(a)

