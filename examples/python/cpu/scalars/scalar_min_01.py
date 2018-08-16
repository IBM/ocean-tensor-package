import pyOcean_cpu as ocean

v1 = [1,2,3,1, ocean.nan, 2+3j, 4-5j]
v2 = [3,2,1,1, 1,         2+2j, 4-6j]

dtypes = [ocean.int8, ocean.uint8, ocean.float, ocean.chalf, ocean.cdouble]

for dtype in dtypes :
   print("\n========= Scalar %s =========" % (dtype.name))
   a1 = [dtype(v) for v in v1]
   a2 = [dtype(v) for v in v2]
   print(a1)
   print(a2)
   print("\n------ min(a1,a2), min(a2,a1), max(a1,a2) ------")
   print([ocean.min(x1,x2) for (x1,x2) in zip(a1,a2)])
   print([ocean.min(x1,x2) for (x1,x2) in zip(a2,a1)])
   print([ocean.max(x1,x2) for (x1,x2) in zip(a1,a2)])

   print("\n------ fmin(a1,a2), fmin(a2,a1), fmax(a1,a2) ------")
   print([ocean.fmin(x1,x2) for (x1,x2) in zip(a1,a2)])
   print([ocean.fmin(x1,x2) for (x1,x2) in zip(a2,a1)])
   print([ocean.fmax(x1,x2) for (x1,x2) in zip(a1,a2)])

