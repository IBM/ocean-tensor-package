import pyOcean_cpu as ocean
import sys


def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

idx = ocean.index[[[0,2],[1,1],[2,-1]]]
failTest("idx.bind([4,2])")

