import pyOcean_cpu as ocean

a = ocean.asTensor([1,1,1,1,0,1])
print(ocean.all(a))
