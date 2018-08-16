import pyOcean_cpu as ocean
import numpy as np
import pyOceanNumpy


a = ocean.asTensor([]);             print(a)
a = ocean.asTensor(1);              print(a)
a = ocean.asTensor(2, ocean.int16); print(a)
a = ocean.asTensor(np.float32(3));  print(a)
a = ocean.asTensor(np.int8(4));     print(a)

a = ocean.asTensor([[1,2,3],[4,5,6]]);  print(a)
a = ocean.asTensor([[1,2,3],[4,5,6.]]); print(a)

t = np.asarray([[1,2],[3,4]],np.int32)
print(t)

a = ocean.asTensor(t);              print(a)
a = ocean.asTensor(t, ocean.float); print(a)

# Combination of tensors
b = ocean.asTensor([t,a, [(1, 1.5),[np.float(3),ocean.asTensor(4)]]], ocean.double);
print(b)

