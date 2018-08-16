import pyOcean_cpu as ocean

# Start with a row-major tensor
a = ocean.tensor([3,4],'c',ocean.int64)
a.T.copy(range(a.nelem))

print(a)
print(a.flatten())
print(a.flatten('c'))
print(a.flatten('a'))
print(a.flatten('k'))


# Start with a column-major tensor (default data type)
a = ocean.tensor([3,4],'f')
a.copy(range(a.nelem))

print(a)
print(a.flatten())
print(a.flatten('c'))
print(a.flatten('a'))
print(a.flatten('k'))

# Start with an unordered tensor
a = ocean.tensor([3,4],[2,6])
a.T.copy(range(a.nelem))

print(a)
print(a.flatten())
print(a.flatten('c'))
print(a.flatten('a'))
print(a.flatten('k'))


