import pyOcean_cpu as ocean

a = ocean.arange(15).reshape([5,3])

print(a + 100)
print(a.T + 100)

a += 100
print(a)

b = a.T
b += 100
print(a)


