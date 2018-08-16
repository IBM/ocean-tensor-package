import pyOcean_cpu as ocean

a = ocean.cfloat(1+2j)
b = ocean.cfloat([1,2,3+2j])

print(b)

print("\n--------- Scalar A times tensor B ---------")
print(ocean.multiply(a,b))
print(ocean.multiply(a,'C',b))
print(ocean.multiply(a,b,'C'))
print(ocean.multiply(a,'C',b,'C'))

print("\n--------- Tensor B times scalar A ---------")
print(ocean.multiply(b,a))
print(ocean.multiply(b,'C',a))
print(ocean.multiply(b,a,'C'))
print(ocean.multiply(b,'C',a,'C'))

