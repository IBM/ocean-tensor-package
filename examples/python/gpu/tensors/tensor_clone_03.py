import ocean

a = ocean.asTensor([1,2,3])

v = a.clone([ocean.cpu, ocean.gpu[0]])
for tensor in v :
   print(tensor)
