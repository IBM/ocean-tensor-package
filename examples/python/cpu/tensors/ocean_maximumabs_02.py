import pyOcean_cpu as ocean

for datatype in [ocean.bool, ocean.int8, ocean.uint8, ocean.int32, ocean.half, ocean.float, ocean.chalf, ocean.double] :
   print("###### %s ######" % datatype.name)
   A = ocean.asTensor([1,-3,2],datatype);
   B = A.clone()
   B.byteswap()
   print(ocean.asTensor([A,B]))
   print(">>> %s %s %s %s" % (ocean.maximumAbs(A),
                              ocean.maximumAbs(B),
                              ocean.maximumAbs([A,B]),
                              ocean.minimumAbs(ocean.asTensor([-1,-3,2],datatype))))
   print("")
