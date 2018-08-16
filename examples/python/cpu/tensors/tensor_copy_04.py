import pyOcean_cpu as ocean

a = ocean.tensor([2,3],[0,ocean.float.size],ocean.float)
a.T.copy([1,2,3])
print(a)

