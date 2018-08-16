import pyOcean_cpu as ocean

a = ocean.tensor([2,3,4,5])
print(a.size)
print(a.strides)

print("\nPermute axes [0,3,2,1]")
b = a.permuteAxes([0,3,2,1])
print(b.size)
print(b.strides)

print("\nIn-place permute axes [1,2,3,0]")
a.permuteAxes([1,2,3,0],True)
print(a.size)
print(a.strides)


a = ocean.tensor([])
a.fill(3)
print(a.size)

b = a.permuteAxes([])
print(b.size)

