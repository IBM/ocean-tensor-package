import ocean

a = ocean.tensor([3,6],[0,1],ocean.double, ocean.cpu)
b = ocean.tensor([3,6],ocean.float, ocean.gpu[0])

a.fill(1)
b.fill(2)

print(a)
print(b)

print("--------------------------------------------------------------")

b.copy(a)

print("--------------------------------------------------------------")
print(a)
print(b)


