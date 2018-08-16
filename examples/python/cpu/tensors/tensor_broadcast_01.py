## Tensor.broadcastLike - broadcasting of tensors

import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

a = ocean.tensor([2])
b = ocean.tensor([2,3])
a.fill(10)
c = a.broadcastLike(b)

print(a)
print(c)
print(c.strides)

c.fill(9)

print(a)
print(c)
print(c.strides)

d = a.shallowCopy()
d.broadcastLike(b,True)
c.detach()
c.fill(8)

print(d)
print(c)
print(c.strides)

print("Automatic broadcast: %s" % ocean.getAutoBroadcast())

b.copy(a)
print(b)

ocean.setAutoBroadcast(False)
print("Automatic broadcast: %s" % ocean.getAutoBroadcast())

failTest("b.copy(a)")

