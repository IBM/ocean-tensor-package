# Construction of tensors from storage
import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()

A = ocean.asTensor(range(12), ocean.int8)
S = A.storage
print(S)

T = ocean.tensor(S)
print(T)

T = ocean.tensor(S,0,[3,4])
print(T)

T = ocean.tensor(S,0,[3,4],[1,3])
print(T)

T = ocean.tensor(S,2,[10]);
print(T)

T = ocean.tensor(S,2,[3,3]);
print(T)

T = ocean.tensor(S,0,[4,5],[1,2])
print(T)

T = ocean.tensor(S,0,[4,9],[1,1])
print(T)

# The tensor extent can be smaller than the storage size
T = ocean.tensor(S,1,[4,6],[1,1])
print(T)


failTest("ocean.tensor(S,-1,[12])")
failTest("ocean.tensor(S,0,[4,16],[1,1])")
