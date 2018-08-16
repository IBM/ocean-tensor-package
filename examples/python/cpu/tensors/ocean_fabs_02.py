import pyOcean_cpu as ocean

device = ocean.cpu

dtypes = [ocean.int8, ocean.int16, ocean.float, ocean.chalf, ocean.cdouble]
otypes = [ocean.float, ocean.uint8, ocean.double, ocean.int32, ocean.chalf]

print("\n========= fabs =========")

for (dtype, otype) in zip(dtypes, otypes) :
   a = ocean.asTensor([-3-4j], dtype, device)
   b = ocean.tensor([1], otype, device)
   ocean.fabs(a,b)
   print(b)


print("\n========= absolute =========")
for (dtype, otype) in zip(dtypes, otypes) :
   a = ocean.asTensor([-3-4j], dtype, device)
   b = ocean.tensor([1], otype, device)
   ocean.absolute(a,b)
   print(b)

