import pyOcean_cpu as ocean

a = ocean.chalf(1+2j)
print(bool(a))
a = ocean.chalf(0+1j)
print(bool(a))
a = ocean.chalf(0+0j)
print(bool(a))

a = ocean.chalf(1+2j)
print(int(a))
print(long(a))
print(float(a))
print(complex(a))
