import pyOcean_cpu as ocean

a = ocean.arange(15).reshape(3,5)
print(a)

idx = ocean.index[[[0,2],[1,1],[2,-1]]]
print(a[idx])

