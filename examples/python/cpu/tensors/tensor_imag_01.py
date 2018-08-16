import pyOcean_cpu as ocean

A = ocean.arange(24,ocean.float).reshape([2,4,3])
print(A)
print(A.imag)
print(A.imag.strides)

A = ocean.asTensor([1+2j,3+4j,5+6j],ocean.chalf)
r = A.imag
print(r)
r.fill(7)
print(A)



