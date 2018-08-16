import pyOcean_cpu as ocean

# Type casting applied to storage objects
s = ocean.asTensor([1,2,3]).storage
print(s)

t = ocean.int8(s)
print(t)

ocean.float(s, True)
print(s)

