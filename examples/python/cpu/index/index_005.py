## Creation of indices
import pyOcean_cpu as ocean
import sys


def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


print("### Indices")
failTest("ocean.index[[1.5,2.6]]")
failTest("ocean.index[[1.5,2.6+3j]]")
failTest("ocean.index[ocean.zeros([3,3,3],ocean.int64)]")
failTest("ocean.index[ocean.zeros([100,3],ocean.int64)]")
failTest("ocean.index[[ocean.uint64.max]]")

print("\n### Range")
failTest("ocean.index[(1+2j):]")
failTest("ocean.index[1:(1+2j)]")
failTest("ocean.index[1:2:(1+2j)]")
failTest("ocean.index[ocean.uint64.max:]")
failTest("ocean.index[:ocean.uint64.max]")
failTest("ocean.index[1::ocean.uint64.max]")
failTest("ocean.index[4:6:0]")
