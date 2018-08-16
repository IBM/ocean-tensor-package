import pyOcean_cpu as ocean


a = ocean.zeros([5,5,5])
a.permuteAxes([1,2,0],True)
print(a.strides)

a[:,[1,2],:] = ocean.arange(50).reshape([5,2,5])
print(a)
