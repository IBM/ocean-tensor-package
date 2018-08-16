import pyOcean_cpu as ocean

a = ocean.asTensor([[1,2],[3,4]],'r',ocean.int16)
b = ocean.asTensor([[1,2],[3,4]],'r',ocean.int16)

print(a)
print(b)

print(ocean.multiply(a,b))
print(ocean.multiply(a,'T',b))
print(ocean.multiply(a,b,'T'))
print(ocean.multiply(a,'T',b,'T'))


