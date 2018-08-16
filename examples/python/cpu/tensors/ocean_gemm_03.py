import pyOcean_cpu as ocean
a = ocean.asTensor([[1,2],[3,4]],'r',ocean.float)
b = ocean.arange(6,ocean.float).reshape([2,3])

print(a)
print(b)

print("\na.T *b")
print(a.T * b)

print("\na * b")
print(a * b)

