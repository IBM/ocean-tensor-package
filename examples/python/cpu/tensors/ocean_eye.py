import pyOcean_cpu as ocean

a = ocean.eye(3)
print(a)

a = ocean.eye(3,ocean.int16)
print(a)

a = ocean.eye(3,ocean.cpu)
print(a)

a = ocean.eye(2,3,ocean.double)
print(a)

a = ocean.eye(2,4,2)
print(a)

a = ocean.eye(3,4,-1,ocean.cpu)
print(a)

