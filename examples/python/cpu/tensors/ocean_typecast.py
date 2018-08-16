import pyOcean_cpu as ocean

t = [[ocean.bool, ocean.int8, ocean.int16, ocean.int32, ocean.int64],
     [ocean.uint8, ocean.uint16, ocean.uint32, ocean.uint64],
     [ocean.half, ocean.float, ocean.double, ocean.chalf, ocean.cfloat, ocean.cdouble]]

for s1 in t :
   for s2 in t :
      s = "%17s" % ""
      for v in s2 :
         s = s + ("%15s" % v.name)
      print(s)
      s = "%17s" % ""
      for v in s2 :
         s = s + ("%15s" % "--------------")
      print(s)
      for u in s1 :
         s = "%-15s |" %  u.name
         for v in s2 :
            a = ocean.asTensor([u(1),v(1)])
            s = s + ("%15s" % a.dtype.name)
         print(s)
      print("")

      
