import pyOcean_cpu as ocean

a = ocean.int8(0);
print(a)
a.imag = 3
print(a)
a.real = -1
print(a)
print(a.dtype)

a = ocean.float(3);
a.imag = ocean.float(2)
print(a)
print(a.dtype)

