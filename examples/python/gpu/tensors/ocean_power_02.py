import ocean

device = ocean.gpu[0]

print("\n========= Tensor and tensor =========")

dtypes = [ocean.int8, ocean.int32, ocean.float, ocean.cfloat];

for dtype in dtypes :
   print("\n------ %s (%s) ------" % (dtype.name, device.name))
   print(ocean.asTensor([1,2,3,4],'r',dtype,device) ** [0,1,2])
   print(device([1,2,3]) ** ocean.asTensor([1,2,3,4],'r',dtype,device))
   print(ocean.asTensor([1,2,3,4],'r',dtype,device) ** [0,0.5,1])
   print(device([1.0,2.0,3.0]) ** ocean.asTensor([1,2,3,4],'r',dtype,device))

