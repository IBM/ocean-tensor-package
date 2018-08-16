import pyOcean_cpu as ocean

s = ocean.scalar([1])
print(s)

s = ocean.scalar(ocean.int8(3), ocean.double)
print(s)

s = ocean.scalar()

