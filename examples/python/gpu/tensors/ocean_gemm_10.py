import ocean

a = ocean.zeros([0,5],ocean.float,ocean.gpu[0])
b = ocean.ones([5],ocean.float,ocean.gpu[0])
c = a * b
print(c)

