import pyOcean_cpu as ocean

import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


# Create a scalar
s = ocean.int8(9)
print(s)
print(type(s))

# Casting with device gives tensor output
print(ocean.cpu(s))

# In-place casting with device gives an error
failTest("ocean.cpu(s,True)")

