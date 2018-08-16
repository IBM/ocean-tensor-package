import pyOcean_cpu as ocean

# Type casting applied to tensor objects
a = ocean.asTensor([1,2,3])
print(a)

b = ocean.int8(a)
print(b)

ocean.float(a, True)
print(a)

