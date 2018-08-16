import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


a = ocean.asTensor([[1,2,0,0,3],[0,4,5,0,0]],'R')
print(a)
print(ocean.find(a))

a = ocean.tensor([])
failTest("ocean.find(a)")

