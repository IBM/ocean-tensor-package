import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

print(ocean.arange(250,256,1,ocean.uint8))
print(ocean.arange(0,2,0.2,ocean.bool))
print(ocean.arange(5,ocean.cdouble))

#failTest("ocean.arange(110,310,50,ocean.uint8)")
#failTest("ocean.arange(28,130,20,ocean.int8)")

