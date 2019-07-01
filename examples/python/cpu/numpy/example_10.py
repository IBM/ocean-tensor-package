import pyOcean_cpu as ocean
import numpy as np
import pyOceanNumpy

import sys

try:
   long
except NameError:
   long = int

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


print(ocean.getImportTypes())
print(ocean.getExportTypes())


a = ocean.tensor([5])

# Python scalars
a.fill(False)
print(a)
a.fill(1)
print(a)
a.fill(2.0)
print(a)
a.fill(long(3))
print(a)
a.fill(4+3j)
print(a)

# Numpy scalars
a.fill(np.int8(5));    print(a)
a.fill(np.int16(6));   print(a)
a.fill(np.int32(7));   print(a)
a.fill(np.int64(8));   print(a)
a.fill(np.uint8(8));   print(a)
a.fill(np.uint16(9));  print(a)
a.fill(np.uint32(10)); print(a)
a.fill(np.uint64(11)); print(a)
a.fill(np.float(12));  print(a)
a.fill(np.double(13)); print(a)

failTest("a.fill('a')")

