import ocean

c = ocean.arange(18, ocean.float, ocean.cpu).reshape([3,6])
d = ocean.tensor([18], ocean.double, ocean.gpu[0])

c.byteswap()
d.fill(2)

print(c)
print(d)

print("--------------------------------------------------------------")

d.copy(c)
d.sync()

print("--------------------------------------------------------------")
print(c)

print(d.storage)
print(d)
