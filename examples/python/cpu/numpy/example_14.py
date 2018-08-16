import pyOcean_cpu as ocean
import pyOceanNumpy

a = ocean.asTensor([1,2,3])
a.readonly = True
b = a.convertTo('numpy')

print(b.flags['WRITEABLE'])

