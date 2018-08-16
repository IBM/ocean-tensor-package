import ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

A = ocean.asTensor([[1+2j,3+1j],[1,2+2j]],ocean.gpu[0])
print(A)
print(ocean.maximumAbs(A))

failTest("ocean.maximumAbs([[]])")

