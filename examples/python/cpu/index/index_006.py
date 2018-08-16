## Creation of index view
import pyOcean_cpu as ocean
import sys


def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


print("## Index view")
idx = ocean.index[3,4]
a = ocean.tensor([3])
failTest("a[idx]")
