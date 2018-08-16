import pyOcean_cpu as ocean

s = ocean.storage(10, ocean.float, ocean.cpu)

a = ocean.tensor(s, 0, [2,3])
b = ocean.tensor(s, 0, [6])

a.copy(ocean.asTensor(range(6),a.dtype))
b.copy(a)

print(a)
print(b)

