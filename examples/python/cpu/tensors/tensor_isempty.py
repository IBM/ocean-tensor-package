import pyOcean_cpu as ocean

a = ocean.tensor([0])
b = ocean.tensor([2,3,0])
c = ocean.tensor([])

c.fill(2)

print(a)
print(b)
print(c)

print("a.isEmpty()  : %s" % (a.isEmpty()))
print("b.isEmpty()  : %s" % (b.isEmpty()))
print("c.isEmpty()  : %s" % (c.isEmpty()))
print("c.isScalar() : %s" % (c.isScalar()))
