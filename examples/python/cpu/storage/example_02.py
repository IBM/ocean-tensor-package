import pyOcean_cpu as ocean

s = ocean.storage(10,ocean.int8)
s.dealloc()
print(s)

s = ocean.storage(0,ocean.int8)
print(s)
