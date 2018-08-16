import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


# Unsigned integer (8)
a = ocean.arange(5,ocean.uint8)
print(ocean.sqrt(a))


# Different types
types = [ocean.int16, ocean.half, ocean.float, ocean.cdouble]
for t in types :
   a = ocean.linspace(-5,5,11,t)
   print(ocean.sqrt(a))


# Result in unsigned integer
a = ocean.arange(-5,5,ocean.int8)
b = ocean.tensor(a.size, ocean.uint8)
ocean.sqrt(a,b)
print(b)
