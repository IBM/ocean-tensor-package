import pyOcean_cpu as ocean

a = ocean.arange(6)

print(a.reshape([2,3]))
print(a.reshape(2,3))
a.reshape(2,3,True)
print(a)
a.reshape([3,2],True)
print(a)

