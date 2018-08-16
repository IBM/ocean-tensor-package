import pyOcean_cpu as ocean

import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

a = ocean.tensor([5,6])
a.copy(range(a.nelem))

failTest("a.slice()")
failTest("a.slice(1)")
failTest("a.slice(-1,1,2)")
failTest("a.slice(2,1,2)")
failTest("a.slice(0,-1,2)")
failTest("a.slice(0,10,1)")
failTest("a.slice(0,4,-1)")
failTest("a.slice(0,4,2)")

