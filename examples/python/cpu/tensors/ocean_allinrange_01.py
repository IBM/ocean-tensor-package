import pyOcean_cpu as ocean

for dtype in [ocean.int8, ocean.half, ocean.cfloat] :
   print("========= (%s) =========" % dtype.name)
   a = ocean.asTensor([1,2,3],dtype)

   print([ocean.allLE(a,3), True])
   print([ocean.allLT(a,3), False])
   print([ocean.allLE(a,3.1), True])
   print([ocean.allLT(a,3.01), True])
   print([ocean.allLE(a,100000), True])
   print([ocean.allLT(a,ocean.inf), True])
   print([ocean.allLE(a,-ocean.inf), False])
   print([ocean.allLE(a,1), False])
   print([ocean.allLE(a,-1000),False])

   print([ocean.allGE(a,1),True])
   print([ocean.allGT(a,1),False])
   print([ocean.allGE(a,0.99),True])
   print([ocean.allGT(a,0.99),True])
   print([ocean.allGT(a,-10000),True])
   print([ocean.allGE(a,-ocean.inf),True])

   print([ocean.allInRange(a,1,3),True])
   print([ocean.allInRange(a,1.1,3),False])
   print([ocean.allInRange(a,1,3.1),True])
   print([ocean.allInRange(a,1.1,2.9),False])

   print([ocean.allInRange(a,1,False,3),False])
   print([ocean.allInRange(a,1,3,False),False])



