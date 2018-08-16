import ocean

a = ocean.tensor([2,6],[0,0],ocean.gpu[0])
b = ocean.tensor([2,6],[0,1],ocean.gpu[0])
c = ocean.tensor([2,2,3],[1,0,0],ocean.gpu[0])
d = ocean.tensor([2,6],[0,0],ocean.gpu[0])

a.fill(1)
b.fill(2)
c.fill(3)
d.fill(4)

print(a)
print(b)
print(c)
print(d)


print("\n--------- Copy ---------\n")

b.copy(a)
c.copy(a)
d.copy(a)

print(a)
print(b)
print(c)
print(d)

print("\n--------- Storage ---------\n")

print(a.storage)
print(b.storage)
print(c.storage)
print(d.storage)

