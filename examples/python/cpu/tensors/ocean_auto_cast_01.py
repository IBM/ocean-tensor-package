## Automatic type case and broadcast
import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


a = ocean.int16([1,2,3])
b = ocean.float([1])

print(a+b)

ocean.setAutoTypecast(False)
print(a+3)

failTest("a+3.2")

print(b+3)
print(b+3.2)
failTest("a+b")


ocean.setAutoBroadcast(False)
print(a+3)
failTest("a+[3]")

