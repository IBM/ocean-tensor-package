import pyOcean_cpu as ocean

dtypes = [ocean.bool, ocean.uint8, ocean.int8, ocean.int64]
shifts = [0,1,5,-1]

for dtype in dtypes :
   print("\n======== Scalar %s =========" % (dtype.name))
   values = [dtype(v) for v in [0,1,8]]
   print(values)
   print(shifts)
   print("\n------ Shift left -----")
   for shift in shifts :
      print([v << shift for v in values])
   print("\n------ Shift right -----")
   for shift in shifts :
      print([v >> shift for v in values])

