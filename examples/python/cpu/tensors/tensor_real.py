import pyOcean_cpu as ocean

A = ocean.asTensor([1,2,3],ocean.float)
print(A.real)

A = ocean.asTensor([1+2j,3+4j,5+6j],ocean.chalf)
r = A.real
print(r)
r.fill(7)
print(A)



