import ocean

a = ocean.asTensor([[0,1,2,3],[4,5,6,7],[8,9,10,11]], ocean.float, ocean.gpu[0])
b = ocean.asTensor([[12,13,14,15],[16,17,18,19],[20,21,22,23]], ocean.float, ocean.gpu[1])
print(a)
print(b)

b.flipAxis(1,True)
print(b)

a.copy(b)
print(a.flipAxis(0))
