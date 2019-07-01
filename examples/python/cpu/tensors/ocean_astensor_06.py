import pyOcean_cpu as ocean
import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      print("\n>>> %s" % command)
      exceptionMsg()


a = ocean.asTensor([[1,2],[2,3,4]],0)
print(a)

a = ocean.asTensor([range(3,7),[7,8]],0)
print(a)

a = ocean.asTensor([[[1],[2,3],[4,5,6]],ocean.ones([3,3])],0)
print(a)

failTest("ocean.asTensor([1,[1,2]])")
failTest("ocean.asTensor([[1,2],1])")
failTest("ocean.asTensor([xrange(4),[[1,2],[3,4]]],0)")
failTest("ocean.asTensor([[[1,2],[3,4]],xrange(4)],0)")
failTest("ocean.asTensor([ocean.ones([3]),ocean.ones([3,4])],0)")
failTest("ocean.asTensor([ocean.ones([2]),ocean.ones([3])])")
failTest("ocean.asTensor([ocean.ones([2,2,2]),ocean.ones([2,3,2])])")
failTest("ocean.asTensor([ocean.ones([2,2,2]),ocean.ones([2,2,3])])")

