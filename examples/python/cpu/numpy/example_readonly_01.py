import pyOcean_cpu as ocean
import pyOceanNumpy
import numpy as np

import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()



a = np.ones([3])
a.setflags(write=False)

b = ocean.asTensor(a)
print(b)

failTest("ocean.sqrt(b,b)")


