import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

A = ocean.asTensor([1,2,3])
B = ocean.tensor([2,3], ocean.int8) # Only size information is used

print(A.broadcastLike(B,1))

failTest("A.broadcastLike(ocean.tensor([2,4]))")
failTest("A.broadcastTo([3,2],2)")

print(A.broadcastTo([2,3], 1))
A.broadcastTo([2,3], 1, True)
print(A)
