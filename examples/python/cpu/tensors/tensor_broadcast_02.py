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
B = ocean.tensor([3,2,1], ocean.int8) # Only size information is used
print(A.broadcastLike(B))

failTest("A.broadcastLike(ocean.tensor([2,4]))")

print(A.broadcastTo([3,2]))
A.broadcastTo([3,2,1], True)
print(A)
