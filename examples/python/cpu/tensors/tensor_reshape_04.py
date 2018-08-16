import pyOcean_cpu as ocean

a = ocean.asTensor(range(24))
b = a.reshape([2,3,4])
print(b)

a = ocean.tensor([4,6],[7,1])
a.copy(range(a.nelem))
b = a.reshape([2,3,4])
print(a)
print(b)

b = a.reshape([6,4])
print(b)

a.reshape([6,4],True)
print(a)



