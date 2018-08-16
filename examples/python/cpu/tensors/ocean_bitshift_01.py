import pyOcean_cpu as ocean

device = ocean.cpu
dtypes = [ocean.bool, ocean.int8, ocean.uint8, ocean.uint32]

for dtype in dtypes :
   print("\n========= Tensor %s (%s) =========" % (dtype.name, device.name))
   a = ocean.asTensor([0,1,2,9],'r',dtype,device)
   b = ocean.asTensor([0,1,2,4,6,8],ocean.uint8,device)
   r = ocean.tensor([6,4],dtype,device)
   print(a)
   print(b)
   print(a << b)
   ocean.bitshiftLeft(a,b,r)
   print(r)
   a <<= 1
   print(a)
