import ocean

a = ocean.zeros([5,0],ocean.float,ocean.gpu[0])
b = ocean.ones([0,5],ocean.float,ocean.gpu[0])
c = a * b
print(c)

