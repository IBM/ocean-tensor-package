import ocean

device = ocean.gpu[0]

dtypes = [ocean.bool, ocean.int32, ocean.uint32, ocean.float, ocean.cfloat]

for dtype in dtypes :
   print("\n========= Tensor %s (%s) =========" % (dtype.name, device.name))
   a1 = ocean.asTensor([0,0,1,1],dtype,device)
   a2 = ocean.asTensor([0,1,0,1],dtype,device)
   r1 = ocean.tensor([4],ocean.bool,device)
   r2 = ocean.tensor([4],ocean.int16,ocean.cpu)
   r2.byteswap()

   print(a1)
   print(a2)

   print("\n------ Logical AND ------")
   ocean.logicalAnd(a1,a2,r1)
   ocean.logicalAnd(a1,a2,r2)
   print(ocean.logicalAnd(a1,a2))
   print(r1)
   print(r2)

   print("\n------ Logical OR ------")
   ocean.logicalOr(a1,a2,r1)
   ocean.logicalOr(a1,a2,r2)
   print(ocean.logicalOr(a1,a2))
   print(r1)
   print(r2)

   print("\n------ Logical XOR ------")
   ocean.logicalXor(a1,a2,r1)
   ocean.logicalXor(a1,a2,r2)
   print(ocean.logicalXor(a1,a2))
   print(r1)
   print(r2)

