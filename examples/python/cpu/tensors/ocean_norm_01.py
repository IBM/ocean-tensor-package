import pyOcean_cpu as ocean
import math

a = [[1+2j, 1+3j, 2+0j],[2+1j,-3-4j,0+0j]];

for dtype in [ocean.int8, ocean.half, ocean.double, ocean.chalf, ocean.cfloat] :
   print("========= (%s) =========" % (dtype.name))
   m = ocean.asTensor(a,'r',dtype)
   for p in [0,1,2,3,ocean.inf] :
      v = m.reshape([6]).asPython(); ref = -1;
      if (p == 0) :
         ref = sum([(e != 0) for e in v])
      elif (p == 1) :
         ref = sum([abs(e) for e in v])
      elif (p == 2) :
         ref = math.sqrt(sum([abs(e)**2 for e in v]))
      elif (p == ocean.inf) :
         ref = max([abs(e) for e in v])
      else :
         ref = math.pow(sum([math.pow(abs(e),p) for e in v]), 1/p)

      print("%3.0f  %9.7f  %9.7f" % (p,float(ocean.norm(m,p)),ref))



