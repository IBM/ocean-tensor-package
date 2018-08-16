import pyOcean_cpu as ocean
import numpy as np
import pyOceanNumpy

a = ocean.tensor([5])
s = ocean.tensor([], ocean.float)
s.fill(3);

a.fill([[False]]);   print(a)

a.fill([(1,)]);      print(a)
a.fill([2.0]);       print(a)
a.fill(s);           print(a)
a.fill([s]);         print(a)

s = np.asarray([4])
print(s)
print(type(s))
a.fill(s);           print(a)
a.fill([(s,)]);      print(a)

