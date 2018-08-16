import ocean

a = ocean.arange(18,ocean.cpu).reshape([3,6])
b = ocean.tensor([3,6],ocean.cpu)
c = ocean.tensor([3,6],ocean.gpu[0])

a.byteswap()
b.fill(1);
c.fill(2)

print(a)
print(b)
print(c)


b.copy(a)
c.copy(a)


print("--------------------------------------------------------------")
print(a)
print(b)
print(c)

d = ocean.double(a)
print(d)

d = ocean.gpu[0](a)
print(d)


