import pyOcean_cpu as ocean
import sys

def fail(command) :
   try :
      eval(command)
      print("%s: SUCCEEDED BUT SHOULD HAVE FAILED!" % command)
   except :
      print("%s: Expected error: %s" % (command, str(sys.exc_info()[1])))


a = ocean.tensor([3,4])
b = ocean.tensor([3])
c = ocean.tensor([3,1])

fail("ocean.sum(a,[])")
fail("ocean.sum(a,0,b)")
fail("ocean.sum(a,1,True,b)")
fail("ocean.sum(a,[1,1])")
fail("ocean.sum(a,[-3])")
fail("ocean.sum(a,[2])")
fail("ocean.sum(a,[1,-1])")

