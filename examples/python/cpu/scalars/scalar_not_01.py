import pyOcean_cpu as ocean

dtypes = [ocean.bool, ocean.int8, ocean.uint8]
values = [0,1,-1]

for dtype in dtypes :
   print("\n========= Scalar %s =========" % (dtype.name))
   V = [dtype(v) for v in values]
   print(V)
   print([~v for v in V])
   print([ocean.bitwiseNot(v) for v in V])
   print([ocean.logicalNot(v) for v in V])


