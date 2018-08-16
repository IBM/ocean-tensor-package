import pyOcean_cpu as ocean

a = ocean.zeros([5,0],ocean.float)
b = ocean.ones([0,5],ocean.float)
c = a * b
print(c)

