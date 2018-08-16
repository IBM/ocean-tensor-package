import pyOcean_cpu as ocean

a = ocean.diag([1,2,3])
print(a)

a = ocean.diag([1,2,3],2,ocean.double)
print(a)

a = ocean.diag([1,2,3],-2,ocean.double)
print(a)

a = ocean.diag([],2,ocean.double)
print(a)

v = ocean.asTensor([1,2,3])
a = ocean.diag(v,ocean.cpu)
print(a)

