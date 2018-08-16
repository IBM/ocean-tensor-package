import pyOcean_cpu as ocean

a = ocean.arange(20,ocean.int8)
print(a)

b = ocean.tensor(a.storage,10,[5],[-1])
print(b)

b.zero()
print(b)
print(a)

