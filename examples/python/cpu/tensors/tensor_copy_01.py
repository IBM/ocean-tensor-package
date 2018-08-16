import pyOcean_cpu as ocean

a = ocean.tensor([2,6],[0,0],ocean.cpu)
b = ocean.tensor([2,6],[0,1],ocean.cpu)
c = ocean.tensor([2,6,2],[0,0,1],ocean.cpu)
d = ocean.tensor([2,6],[0,0],ocean.cpu)

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

