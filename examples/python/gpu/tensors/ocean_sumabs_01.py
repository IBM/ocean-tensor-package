import ocean

print(ocean.sumAbs(ocean.gpu[0]([1,-2,3])))

for dtype in [ocean.int8, ocean.int16, ocean.half, ocean.chalf, ocean.cdouble] :
   a = ocean.asTensor([1,-2,5],dtype,ocean.gpu[0])
   print(ocean.sumAbs(a))

