import pyOcean_cpu as ocean

# Type casting applied to scalars
a = ocean.int16(3)
print(a)

b = ocean.int8(a)
print(b)

ocean.float(a, True)
print(a)

