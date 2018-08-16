import pyOcean_cpu as ocean
import sys


def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


print("## Index binding")
idx = ocean.index[[1,2,3]]
idx.bind([5],True)
failTest("idx.bind([5,7],True)")
failTest("idx.bind([6],True)")
idx.bind([5],[8],True)
failTest("idx.bind([5],[4],True)")

idx = ocean.index[1,3]
idx.bind([5,5],True)
failTest("idx.bind([5,6,7])")



print("\n## Scalar")
idx = ocean.index[10]
failTest("idx.bind([5],True)")
idx = ocean.index[1,2,3]
failTest("idx.bind([5,5,2],True)")

print("\n## Indices")
idx = ocean.index[1,[1,2,10]]
failTest("idx.bind([5,5])")

print("\n## Mask")
idx = ocean.index[ocean.zeros([5,6],ocean.bool)]
failTest("idx.bind([5,5])")

print("\n## Ellipsis")
idx = ocean.index[1,...,3,...,5]
failTest("idx.bind([5,5,5])")

