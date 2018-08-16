import pyOcean_cpu as ocean

a = ocean.zeros([5,5])
a[1,...,3] = 99
print(a)
