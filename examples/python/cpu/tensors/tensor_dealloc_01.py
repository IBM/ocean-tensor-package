import pyOcean_cpu as ocean

a = ocean.zeros([3,4],ocean.float)
print(a)
a.dealloc()
print(a)
print(a.storage)

