import ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


s = ocean.arange(10,ocean.int8,ocean.gpu[0]).storage
print(s)

failTest("ocean.tensor(s,1,[1],[2],1, ocean.int16)")


