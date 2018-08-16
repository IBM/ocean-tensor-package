## Cloning of tensors
import ocean

a = ocean.asTensor([[1,2,3],[4,5,6]], 'r', ocean.float)

print(a.clone())
print(a.clone(ocean.gpu[0]))

a = ocean.asTensor([[2,3,4],[5,6,7]], 'r', ocean.double, ocean.gpu[0])
print(a.clone())
print(a.clone(ocean.cpu))

