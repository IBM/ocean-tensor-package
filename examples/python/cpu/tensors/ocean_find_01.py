import pyOcean_cpu as ocean

a = ocean.asTensor([[[1,0,1],[0,1,1]], [[0,0,0],[1,1,0]]],'r',ocean.bool)
print(a)
print(ocean.find(a))

