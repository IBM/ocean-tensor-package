import ocean

A = ocean.asTensor([1,2,3])
B = A.storage
C = ocean.int8(10)

print(ocean.gpu[0](A))
print(ocean.ensure(A,ocean.float,ocean.gpu[0]))
ocean.ensure(A,ocean.half,ocean.gpu[0],True)
print(A)


print(ocean.gpu[0](B))
print(ocean.ensure(B,ocean.int8,ocean.gpu[0]))
ocean.ensure(B,ocean.gpu[0],True)
print(B)


print(ocean.gpu[0](C))
print(ocean.ensure(C,ocean.int16,ocean.cpu))


