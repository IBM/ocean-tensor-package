import pyOcean_cpu as ocean
import numpy as np

def floordivide(a,b) :
   print("---------------------------------------------")
   print("Numpy : %2d floor divide %2d = %d" % (a,b, np.floor_divide(a,b)))
   print("Python: %2d floor divide %2d = %d" % (a,b, a//b))
   print("Ocean : %2d floor divide %2d = %d" % (a,b, ocean.floorDivide(a,b)))
   for dtype in [ocean.int8, ocean.half, ocean.float, ocean.double] :
      z = ocean.floorDivide(dtype(a),dtype(b))
      print("Ocean : %2d floor divide %2d = %s (%s,%s)" % (a,b, z, dtype.name, z.dtype.name))
      A = ocean.asTensor([a],dtype)
      B = ocean.asTensor([b],dtype)
      print(A // B)


floordivide( 4, 3)
floordivide( 4,-3)
floordivide(-4, 3)
floordivide(-4,-3)
floordivide(-2,-3)
floordivide( 2,-3)
floordivide(10, 5)
floordivide(10,-5)
