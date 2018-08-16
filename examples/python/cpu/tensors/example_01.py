## Construction of tensors with strides dimensions
import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


T = ocean.tensor([])
print(T.footer)

T = ocean.tensor([0], ocean.int8)
print(T.footer)

T = ocean.tensor([3,4])
print(T.footer)

T = ocean.tensor([3,5], ocean.int16, ocean.cpu)
print(T.footer)
print(T.strides)

T = ocean.tensor([3,5],[1,3], ocean.int16, ocean.cpu)
print(T.footer)
print(T.strides)

T = ocean.tensor([3,5],[1,3], 1, ocean.int16, ocean.cpu)
print(T.footer)
print(T.strides)


failTest("ocean.tensor([2,-3])")
failTest("ocean.tensor(ocean.int16, ocean.cpu)")
