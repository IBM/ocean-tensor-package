import pyOcean_cpu as ocean
A = ocean.asTensor([[0,0,1],[0,1,1],[1,0,0]],'r',ocean.bool)

print("\n--------- A ---------")
print(A)

print("\n--------- A' ---------")
print(A.T)

print("\n--------- A * A ---------")
print(ocean.multiply(A,A))

print("\n--------- A * A' ---------")
print(ocean.multiply(A,A,'T'))

print("\n--------- A' * A ---------")
print(ocean.multiply(A,'T',A))

print("\n--------- A' * A' ---------")
print(ocean.multiply(A,'T',A,'T'))

