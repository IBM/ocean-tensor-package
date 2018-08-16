import ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

failTest("ocean.tensor([3,4],[4,2],1,ocean.gpu[0])")


