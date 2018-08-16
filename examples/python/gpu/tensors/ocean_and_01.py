import ocean

device = ocean.gpu[0]

dtypes = [ocean.bool, ocean.int8, ocean.uint8, ocean.int32]
v1 = [1,2,3,7,127]
v2 = [0,1,2,4,8,15]

for dtype in dtypes :
   print("\n========= Tensor %s (%s) =========" % (dtype.name, device.name))
   a1 = ocean.asTensor(v1,'r',dtype,device)
   a2 = ocean.asTensor(v2,dtype,device).reshape([len(v2),1])
   r  = ocean.tensor([len(v2),len(v1)],dtype,device)
   print(a1)
   print(a2)
   print("\n------ bitwiseAnd(a1,a2), (a1 & a2), bitwiseAnd(a1,a2,r) ------")
   print(ocean.bitwiseAnd(a1,a2))
   print(a1 & a2)
   ocean.bitwiseAnd(a1,a2,r)
   print(r)

