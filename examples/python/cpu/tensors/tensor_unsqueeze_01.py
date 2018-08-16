import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


a = ocean.arange(12).reshape([3,4])
print(a)

print(a.unsqueeze(0))
print(a.unsqueeze(2))

a.unsqueeze(0,True)
print(a)
a.unsqueeze(2,True)
print(a)

failTest("a.unsqueeze(-1)")
failTest("a.unsqueeze(5)")

