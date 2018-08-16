import pyOcean_cpu as ocean

a = ocean.ones([3,4])
print(a)

a = ocean.ones([3,4],ocean.int8)
print(a)

a = ocean.ones([3,4],ocean.cpu)
print(a)

a = ocean.ones([3,4],ocean.float,ocean.cpu)
print(a)


