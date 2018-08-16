import pyOcean_cpu as ocean

dtypes = [ocean.bool, ocean.int8, ocean.int16, ocean.int32, ocean.int64,
          ocean.uint8, ocean.uint16, ocean.uint32, ocean.uint64,
          ocean.half, ocean.float, ocean.double, ocean.chalf, ocean.cfloat, ocean.cdouble]

for dtype in dtypes :
   print("%-15s %s" % (dtype.name, dtype(12.34 + 56.78j)))

for dtype1 in dtypes :
   v = dtype1(12.34 + 56.78j)
   print([dtype2(v) for dtype2 in dtypes])
