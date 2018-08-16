import pyOcean_cpu as ocean
import numpy as np

def divide(a,b) :
   print("")
   print("Numpy : %2d divide %2d = %d" % (a,b, np.divide(a,b)))
   print("Python: %2d divide %2d = %d" % (a,b, a/b))
   print("Ocean : %2d divide %2d = %d" % (a,b, ocean.divide(a,b)))
   for dtype in [ocean.int8, ocean.half, ocean.float, ocean.double] :
      z = ocean.divide(dtype(a),dtype(b))
      print("Ocean : %2d divide %2d = %s (%s,%s)" % (a,b, z, dtype.name, z.dtype.name))
      A = ocean.asTensor([a],dtype)
      B = ocean.asTensor([b],dtype)
      print(A / B)


divide( 4, 3)
divide( 4,-3)
divide(-4, 3)
divide(-4,-3)
divide(-2,-3)
divide( 2,-3)
divide(10, 5)
divide(10,-5)
