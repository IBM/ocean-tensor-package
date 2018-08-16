import ocean

a = ocean.asTensor([1,1,1,1,0,1], ocean.gpu[0])
print(ocean.all(a))
