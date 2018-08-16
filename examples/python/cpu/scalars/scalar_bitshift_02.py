import pyOcean_cpu as ocean

dtypes = [ocean.bool, ocean.uint8, ocean.int8, ocean.int64]
shifts = [0,1,5,-1]

for dtype in dtypes :
   print("\n======== Scalar %s =========" % (dtype.name))
   values = [dtype(v) for v in [0,1,8]]
   print(values)
   for i in range(len(values)) :
      values[i] <<= 1
   print(values)
   for i in range(len(values)) :
      values[i] >>= 1
   print(values)

