import pyOcean_cpu as ocean

device = ocean.cpu

dtypes = [ocean.int8, ocean.int16, ocean.float, ocean.chalf, ocean.cdouble]

print("\n========= fabs =========")
for dtype in dtypes :
   a = dtype(-3-4j)
   b = ocean.fabs(a)
   print([b, b.dtype])

for dtype in dtypes :
   a = dtype(-3-4j)
   b = ocean.fabs(device([a]))
   print(b)


print("\n========= absolute =========")
for dtype in dtypes :
   a = dtype(-3-4j)
   b = ocean.absolute(a)
   print([b, b.dtype])

for dtype in dtypes :
   a = dtype(-3-4j)
   b = ocean.absolute(device([a]))
   print(b)
