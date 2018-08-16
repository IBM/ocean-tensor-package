import ocean

a = ocean.asTensor([1,2,3],ocean.int8, ocean.cpu)
b = ocean.tensor([3],ocean.int16,ocean.gpu[0])

ocean.square(a,b)
print(b)

a.byteswap()
b.fill(0)
ocean.square(a,b)
print(b)

a = ocean.asTensor([1,2,3],ocean.int8, ocean.gpu[0])
b = ocean.tensor([3],ocean.int16,ocean.cpu)
ocean.square(a,b)
print(b)

b.fill(0)
b.byteswap()
ocean.square(a,b)
print(b)


