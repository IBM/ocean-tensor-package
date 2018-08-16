import pyOcean_cpu as ocean

a = ocean.zeros([0,5],ocean.float)
b = ocean.ones([5],ocean.float)
c = a * b
print(c)

