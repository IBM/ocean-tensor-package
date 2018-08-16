## Index strides
import pyOcean_cpu as ocean

a = ocean.index[3,[4,5,6]]
a.bind([5,10],[1,8],True)
print(a.strides)

