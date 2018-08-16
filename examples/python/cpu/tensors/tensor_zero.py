import pyOcean_cpu as ocean

a = ocean.tensor([3,4])
a.zero()
print(a)


a = ocean.tensor([2,1,4],ocean.chalf)
a.zero()
print(a)



