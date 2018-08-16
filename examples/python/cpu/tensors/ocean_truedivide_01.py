import pyOcean_cpu as ocean
import numpy as np

def truedivide(a,b) :
   print("---------------------------------------------")
   print("Numpy : %2d true divide %2d = %d" % (a,b, np.true_divide(a,b)))
   print("Ocean : %2d true divide %2d = %d" % (a,b, ocean.trueDivide(a,b)))
   for dtype in [ocean.int8, ocean.half, ocean.float, ocean.double] :
      z = ocean.trueDivide(dtype(a),dtype(b))
      print("Ocean : %2d true divide %2d = %s (%s,%s)" % (a,b, z, dtype.name, z.dtype.name))
      A = ocean.asTensor([a],dtype)
      B = ocean.asTensor([b],dtype)
      print(ocean.trueDivide(A,B))


truedivide( 4, 3)
truedivide( 4,-3)
truedivide(-4, 3)
truedivide(-4,-3)
truedivide(-2,-3)
truedivide( 2,-3)
truedivide(10, 5)
truedivide(10,-5)
