import pyOcean_cpu as ocean

a = ocean.asTensor([[1,0,1,0],[0,0,1,1]],'r')
print(a)
print(ocean.all(a,0))

b = ocean.tensor([1,4],ocean.bool)
ocean.any(a,0,True,b)
print(b)


print(ocean.allFinite(a,1))
print(ocean.anyInf(a,0))

