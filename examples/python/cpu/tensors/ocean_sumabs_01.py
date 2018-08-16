import pyOcean_cpu as ocean

print(ocean.sumAbs([1,-2,3]))

for dtype in [ocean.int8, ocean.int16, ocean.half, ocean.chalf, ocean.cdouble] :
   a = ocean.asTensor([1,-2,5],dtype)
   print(ocean.sumAbs(a))

