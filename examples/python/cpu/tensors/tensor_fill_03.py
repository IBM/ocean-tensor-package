import pyOcean_cpu as ocean

a = ocean.tensor([3,2,4],'C')
a.fill(3)
print(a)

a.reverseAxes(True)
a.fill(4)
print(a)
