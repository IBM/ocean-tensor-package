import pyOcean_cpu as ocean

a = ocean.asTensor([1,2,3])
b = ocean.asTensor([0,3,6],'r')

print(a)
print(b)
print(a+b)

