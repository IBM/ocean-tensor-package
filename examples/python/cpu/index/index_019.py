import pyOcean_cpu as ocean

a = ocean.arange(24).reshape([4,6]);
print(a)
print(a[3])
print(a[3,[1,3,5]])
print(a[[0,1,3],3])
