import ocean

device = ocean.gpu[0]

v1 = [1,2,3,1, ocean.nan, 2+3j, 4-5j]
v2 = [3,2,1,1, 1,         2+2j, 4-6j]

dtypes = [ocean.int8, ocean.uint8, ocean.float, ocean.chalf, ocean.cdouble]

for dtype in dtypes :
   print("\n========= %s (%s) =========" % (dtype.name, device.name))
   a1 = ocean.asTensor(v1,dtype,device)
   a2 = ocean.asTensor(v2,dtype,device)
   r1 = ocean.tensor([len(v1)],dtype,device)
   r2 = ocean.tensor([len(v1)],ocean.cfloat,device)
   print(a1)
   print(a2)
   print("\n------ min(a1,a2), min(a2,a1), max(a1,a2) ------")
   print(ocean.min(a1,a2))
   print(ocean.min(a2,a1))
   print(ocean.max(a1,a2))

   print("\n------ fmin(a1,a2), fmin(a2,a1), fmax(a1,a2) ------")
   print(ocean.fmin(a1,a2))
   print(ocean.fmin(a2,a1))
   print(ocean.fmax(a1,a2))

   print("\n------ fmin(a1,a2,r1), fmin(a1,a2,r2) ------")
   ocean.fmin(a1,a2,r1)
   ocean.fmin(a1,a2,r2)
   print(r1)
   print(r2)
