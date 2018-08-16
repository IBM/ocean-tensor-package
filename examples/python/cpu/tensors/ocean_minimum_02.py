import pyOcean_cpu as ocean
import sys

def fail(command) :
   try :
      eval(command)
      print("%s: SUCCEEDED BUT SHOULD HAVE FAILED!" % command)
   except :
      print("%s: Expected error: %s" % (command, str(sys.exc_info()[1])))


a = ocean.tensor([0,3,4])
fail("ocean.minimum(a)")

