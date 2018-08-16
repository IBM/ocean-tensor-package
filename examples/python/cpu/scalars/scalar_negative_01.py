import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


a = ocean.double(2)
print(a)
print(-a)

a = ocean.half(2)
print(a)
print(-a)

a = ocean.cdouble(1+2j)
print(-a)

a = ocean.chalf(1+2j)
print(-a)

a = ocean.bool(1)
print(a)
print(-a)

a = ocean.uint8(1)
print(-a)

ocean.setScalarCastMode(0)
a = ocean.uint8(1)
failTest("-a")


