### Masked indexing, in-place addition and subtraction
import pyOcean_cpu as ocean

a = ocean.arange(25).reshape(5,5)
print(a)
print(a % 2)

print(ocean.find(a % 2))

a[a%2 == 1] = 10 * ocean.arange(12) + 10;
print(a)


a[:,1] += 1
print(a)

a[2] += a[2]
print(a)
