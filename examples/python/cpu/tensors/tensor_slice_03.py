import pyOcean_cpu as ocean

a = ocean.arange(24).reshape([4,6])
print(a.slice(0,2))
print(a.slice(1,3))

