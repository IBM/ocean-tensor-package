import ocean

# Use GPU tensor to index CPU tensor
v = ocean.gpu[0]([1,2,3])
idx = ocean.index[v]
a = ocean.asTensor([5,6,7,8,9],ocean.double)
print(a[idx])
print(a[v])

# Use byte-swapped CPU tensor to index GPU tensor
a = ocean.asTensor([1,2,8],ocean.int32)
a.byteswap()
print(a)
v = ocean.arange(10, ocean.int16, ocean.gpu[0])
idx = ocean.index[a]

