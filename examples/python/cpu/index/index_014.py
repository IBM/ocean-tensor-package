## Index binding
import pyOcean_cpu as ocean

a = [True,True,False];
b = ocean.index[a]
print([b,b.isBound()])
b.bind([3],True)
print([b,b.isBound()])

b.bind([3],[ocean.int64.size],True)
print([b,b.isBound()])

a = ocean.asTensor([1,2,3],ocean.int64)
print(a[b])
