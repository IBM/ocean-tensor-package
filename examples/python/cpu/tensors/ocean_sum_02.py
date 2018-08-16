import pyOcean_cpu as ocean

print(ocean.sum([1,2,3]))

a = ocean.asTensor([True,False,True],ocean.bool)
print(ocean.sum(a))

for dtype in [ocean.int8, ocean.int16, ocean.half, ocean.chalf, ocean.cdouble] :
   a = ocean.asTensor([1,2,5],dtype)
   print(ocean.sum(a))

