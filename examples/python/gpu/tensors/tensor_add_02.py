import ocean

for device in [ocean.cpu, ocean.gpu[0]] :
   print("\n# Boolean on %s" % (device.name))
   a = ocean.asTensor([True, True, False],device)
   print(a)
   print(+a)
   print(a + True)
   a += True
   print(a)

   print("\n# Integer on %s" % (device.name))
   a = ocean.asTensor([-2,-1,0,1,2],ocean.int8,device)
   print(a)
   print(+a)
   print(a + 3)
   a += 3
   print(a)

   print("\n# Float on %s" % (device.name))
   a = ocean.asTensor([-2,-1,0,1,2],ocean.float,device)
   print(a)
   print(+a)
   print(a + 3.0)
   a += 3
   print(a)

