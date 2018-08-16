import ocean

device = ocean.gpu[0]
dtypes = [ocean.int8, ocean.uint8, ocean.int32, ocean.float, ocean.chalf, ocean.cfloat]
values = [-1,0,1,1+2j,0+2j, 0-2j, 0+0j]

for dtype in dtypes :
   print("\n========= Tensor %s =========" % dtype.name)
   a = ocean.asTensor(values,dtype,device)
   b = ocean.tensor(len(values),ocean.int8,device)
   print(a)
   print(ocean.sign(a))
   ocean.sign(a,b)
   print(b)



