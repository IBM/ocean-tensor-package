import ocean

A = ocean.asTensor([1,2,3])
print(ocean.gpu[1](A))
print(ocean.gpu[1](A.storage))
print(ocean.gpu[0]([[1,2,3],[4,5,6.]]))
