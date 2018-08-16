import pyOcean_cpu as ocean

s = ocean.storage(5)
s.asTensor().fill(1)
print(s)

t = ocean.storage(5, ocean.half)
t.copy(s)
print(t)

t.byteswap()
print(t)

s.copy(t)
print(s)

s.asTensor().fill(3)
print(s)

t.copy(s)
print(t)
