import pyOcean_cpu as ocean

a = ocean.tensor([5,6])
a.T.copy(range(a.nelem))

print(a)
print(a.slice(0,1,2))
print(a.slice(1,2,2))
print(a.slice(0,1,0))
