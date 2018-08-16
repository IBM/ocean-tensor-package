import pyOcean_cpu as ocean

a = ocean.zeros([3,4,5])

a[1,2,3] = 4
a[-1,-1,-1] = 99
print(a)

