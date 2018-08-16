import pyOcean_cpu as ocean

a = ocean.tensor([5,6])
a.T.copy(range(a.nelem))

print(a)

v = a.split(0,2)
print(v)

v = a.split(1,2)
print(v)
