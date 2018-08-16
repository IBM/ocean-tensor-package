import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


# Boolean
a = ocean.asTensor([True,False,False])
print(-a)

# Unsigned integer (8)
a = ocean.arange(5,ocean.uint8)
print(-a)


# Different types
types = [ocean.int16, ocean.half, ocean.float, ocean.cdouble]
for t in types :
   a = ocean.linspace(-5,5,11,t)
   print(-a)


# Result in unsigned integer
a = ocean.arange(-5,5,ocean.int8)
b = ocean.tensor(a.size, ocean.uint8)

failTest("ocean.negative(a,b)")
