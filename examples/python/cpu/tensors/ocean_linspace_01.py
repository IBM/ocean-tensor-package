import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

failTest("ocean.linspace(1+60000j,2+70000j,10,ocean.chalf)")
failTest("ocean.linspace(60000+1j,70000+2j,10,ocean.chalf)")
failTest("ocean.linspace(70000+2j,0,10,ocean.chalf)")
failTest("ocean.linspace(1+70000j,0,10,ocean.chalf)")


