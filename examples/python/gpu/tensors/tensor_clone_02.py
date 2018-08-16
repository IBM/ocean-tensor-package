import ocean

t = ocean.tensor([2,3],[0,1], ocean.float)
t.copy(ocean.asTensor([[1,2,3]],'r'))

print(t)
print(t.storage)
print(t.strides)


print("\nClone")
a = t.clone(ocean.gpu[0])
print(a.storage)
print(a.strides)

print("\nReplicate")
b = t.replicate(ocean.gpu[0])
print(b.storage)
print(b.strides)

