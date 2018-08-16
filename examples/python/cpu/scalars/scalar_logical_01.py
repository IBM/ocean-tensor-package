import pyOcean_cpu as ocean

v1 = [0,0,1,1]
v2 = [0,1,0,1]

dtypes = [ocean.bool, ocean.int8, ocean.float, ocean.cdouble]

for dtype in dtypes :
   print("\n========= Scalar %s =========" % (dtype.name))
   print([dtype(v) for v in v1])
   print([dtype(v) for v in v2])

   print("\n------ Logical AND, OR, XOR ------")
   print([ocean.logicalAnd(dtype(v),dtype(w)) for (v,w) in zip(v1,v2)])
   print([ocean.logicalOr(dtype(v),dtype(w)) for (v,w) in zip(v1,v2)])
   print([ocean.logicalXor(dtype(v),dtype(w)) for (v,w) in zip(v1,v2)])



