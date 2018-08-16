import pyOcean_cpu as ocean

a = ocean.full([2,3],1.0)
b = ocean.full([3,2],2.0)

print(ocean.asTensor([a,b,[[3]]], 0, ocean.int16))

