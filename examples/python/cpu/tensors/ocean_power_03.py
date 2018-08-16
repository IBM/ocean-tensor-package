import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


device = ocean.cpu

print("\n========= Tensor and tensor =========")

a = ocean.asTensor([1,2,3,4],'r',ocean.float,device);
b = ocean.asTensor([0,0.5,1,2],ocean.float,device);

print(a**b)
print(pow(a,b))
print(ocean.power(a,b))

c = ocean.tensor([4,4],ocean.float,device)
ocean.power(a,b,c)
print(c)

c = ocean.tensor([4,4],ocean.cdouble,device)
ocean.power(a,b,c)
print(c)

c = ocean.tensor([4,4],ocean.int16,device)
ocean.power(a,b,c)
print(c)

a **= ocean.asTensor([1,1,1,0.5],'r',ocean.float,device)
print(a)

failTest("pow(a,b,3)")

