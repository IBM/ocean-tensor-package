import pyOcean_cpu as ocean

A = ocean.arange(12,ocean.float)
print(A)

A.byteswap()
print(A)

print(A.reshape([3,4]))

A.reshape([3,4],True)
print(A)


print(A.isContiguous())
print(A.isLinear())

B = A.swapAxes(0,1)
print(B)
print(B.isContiguous())
print(B.isLinear())


B = A.T
print(B)
print(B.isContiguous())
print(B.isLinear())

