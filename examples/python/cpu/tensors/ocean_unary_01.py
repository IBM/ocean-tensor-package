import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

a = ocean.asTensor([1,2,3],ocean.int8)
b = ocean.tensor([3],ocean.uint16)

ocean.square(a,b)
print(b)

b.fill(0)
b.byteswap()
ocean.square(a,b)
print(b)

a.byteswap()
b.fill(0)
ocean.square(a,b)
print(b)

b.byteswap()
b.fill(0)
ocean.square(a,b)
print(b)

# Cannot have read-only output
b.readonly = True
failTest("ocean.square(a,b)")

# Cannot have a tensor with repeats
b = ocean.tensor([3],[0],ocean.int8)
failTest("ocean.square(a,b)")
   
