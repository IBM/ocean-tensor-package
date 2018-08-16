## Set index with byteswapped data
import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

# Tensor is byteswapped
A = ocean.zeros([4,5])
A.byteswap()

A[2:4,1:3] = 3
print(A)

B = ocean.arange(2*2).reshape(2,2) + 1
print(B)

A.fill(0)
A[1:3,2:4] = B
print(A)

B.byteswap()
A.fill(0)
A[2:0:-1,3:1:-1] = B
print(A)

A = ocean.zeros([4,5])
A[[0,3],[3,0]] = B
print(A)

A.readonly = True
try :
   A[[0,3],[3,0]] = 3
except :
   exceptionMsg()

try :
   A[1,3] = 5
except :
   exceptionMsg()

try :
   A[[0,3],[3,0]] = B
except :
   exceptionMsg()

