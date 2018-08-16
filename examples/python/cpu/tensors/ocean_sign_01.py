import pyOcean_cpu as ocean


dtypes = [ocean.int8, ocean.uint8, ocean.int32, ocean.float, ocean.chalf, ocean.cfloat]
values = [-1,0,1,1+2j,0+2j, 0-2j, 0+0j]


for dtype in dtypes :
   print("\n========= Scalar %s =========" % dtype.name)
   print([dtype(value) for value in values])
   print([ocean.sign(dtype(value)) for value in values])

for dtype in dtypes :
   print("\n========= Tensor %s =========" % dtype.name)
   a = ocean.asTensor(values,dtype)
   b = ocean.tensor(len(values),ocean.int8)
   print(a)
   print(ocean.sign(a))
   ocean.sign(a,b)
   print(b)

print("\n========= Exceptions =========")
s = ocean.nan
print(s)
print(ocean.sign(s))
print(ocean.sign([s]))


s = ocean.cfloat(0)
s.imag = ocean.nan
print(s)
print(ocean.sign(s))
print(ocean.sign([s]))

s.real = ocean.nan
s.imag = 0
print(s)
print(ocean.sign(s))
print(ocean.sign([s]))



