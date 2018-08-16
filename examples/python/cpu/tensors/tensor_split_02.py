import pyOcean_cpu as ocean

a = ocean.tensor([5,6])
a.T.copy(range(a.nelem))

print(a)

v = a.split(0,2,True)
v[0].fill(1)
v[1].fill(2)
print(a)

v = a.split(0,2,False)
v[0].fill(3)
v[1].fill(4)
print(a)
