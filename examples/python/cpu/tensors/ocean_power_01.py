import pyOcean_cpu as ocean

device = ocean.cpu

print("\n========= Scalar and tensor =========")

dtypes = [ocean.int8, ocean.int32, ocean.float, ocean.cfloat];

for dtype in dtypes :
   print("\n------ %s (%s) ------" % (dtype.name, device.name))
   print(ocean.asTensor([1,2,3,4],dtype,device) ** 2)
   print(2 ** ocean.asTensor([1,2,3,4],dtype,device))
   print(ocean.asTensor([1,2,3,4],dtype,device) ** 0.5)
   print(2.0 ** ocean.asTensor([1,2,3,4],dtype,device))

