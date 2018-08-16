import ocean

A = ocean.asTensor([1,2,3], ocean.float, ocean.gpu[0])
print(ocean.int8(A))
print(ocean.cdouble(A.storage))
print(ocean.half([[1,2,3],[4,5,6.]]))
print(ocean.float(1.2))

