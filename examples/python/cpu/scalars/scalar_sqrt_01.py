import pyOcean_cpu as ocean
import sys

a = ocean.double(2)
print(ocean.sqrt(a))

a = ocean.half(2)
print(ocean.sqrt(a))

a = ocean.cdouble(1+2j)
print(ocean.sqrt(a))


print(ocean.sqrt(2.0))
print(ocean.sqrt(1-2j))

