import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

failTest("ocean.linspace(1,70000,10,ocean.half)")
failTest("ocean.linspace(-10,20,ocean.uint8)")
failTest("ocean.linspace(0,260,10,ocean.int8)")
failTest("ocean.linspace(260,ocean.int8)")

