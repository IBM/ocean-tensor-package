import pyOcean_cpu as ocean

s = ocean.asTensor([0,1,2], ocean.bool)
print(s.asPython())

s = ocean.asTensor([[1,2,3],[4,5,6]], ocean.float, ocean.cpu)
print(s.asPython())

s = ocean.asTensor(7, ocean.float, ocean.cpu)
print(s.asPython())

