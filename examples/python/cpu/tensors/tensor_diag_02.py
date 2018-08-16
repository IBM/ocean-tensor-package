import pyOcean_cpu as ocean

a = ocean.arange(9).reshape([3,3])
a.byteswap()
a.readonly = True
print(a)

b = a.diag()
print(b)

