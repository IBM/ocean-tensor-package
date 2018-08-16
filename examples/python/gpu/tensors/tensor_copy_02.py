import ocean

a = ocean.gpu[0](12345)

b = ocean.tensor([], ocean.float, ocean.gpu[0])
b.copy(a)
print(b)

b = ocean.double(b)
print(b)

b = ocean.cdouble(b)
print(b)

b.copy(54321)
print(b)

b.copy(ocean.gpu[0](1.2345))
print(b)

