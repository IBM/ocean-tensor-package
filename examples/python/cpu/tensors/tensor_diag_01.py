import pyOcean_cpu as ocean
a = ocean.tensor([5,5])
a.fill(0)
print(a)
a.diag().fill(3)
a.fliplr().diag().fill(5)
print(a)
