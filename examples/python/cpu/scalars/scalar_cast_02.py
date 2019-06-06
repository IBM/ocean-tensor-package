import pyOcean_cpu as ocean

try:
  long
except NameError:
  long = int

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
