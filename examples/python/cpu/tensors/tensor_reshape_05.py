import pyOcean_cpu as ocean

A = ocean.arange(12,ocean.float)
print(A)

A.byteswap()
print(A)

print(A.reshape([3,4]))

A.reshape([3,4],True)
print(A)

