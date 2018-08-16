import ocean

device = ocean.gpu[0]
dtypes = [ocean.bool, ocean.int8, ocean.uint8]

for dtype in dtypes :
   print("\n========= Tensor %s (%s) =========" % (dtype.name, device.name))
   a = ocean.asTensor([0,1,-1],dtype,device)
   print(a)
   print(ocean.bitwiseNot(a))
   print(~a)
   print(ocean.logicalNot(a))

   r = ocean.tensor(3,dtype,device)
   ocean.bitwiseNot(a,r)
   print(r)
   ocean.logicalNot(a,r)
   print(r)

