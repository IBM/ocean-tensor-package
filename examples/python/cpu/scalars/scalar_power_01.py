import pyOcean_cpu as ocean

import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()



a = ocean.int8(3)
print(a ** 2)
print(pow(a,2))
a**= 2
print(a)

failTest("pow(a,2,3)")

print("\n------ Floating point ------")
a = ocean.float(4)
b = ocean.int16(3)
print(a ** b)
print(pow(a,b))
a **= b
print(a)

print("\n------ Square root ------")
a = ocean.half(4)
b = a ** ocean.half(0.5)
print([b,b.dtype.name])
print(a ** 0.5)
print(ocean.power(a,0.5))
print(pow(a,0.5))



