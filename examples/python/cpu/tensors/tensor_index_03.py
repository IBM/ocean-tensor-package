## Range indexing #2
import pyOcean_cpu as ocean
import sys

def success(command) :
   print("%s = " % command)
   print(eval(command))
   print("-----------------------------------")

def fail(command) :
   try :
      eval(command)
      print("%s: SUCCEEDED BUT SHOULD HAVE FAILED!" % command)
   except :
      print("%s: Expected error: %s" % (command, str(sys.exc_info()[1])))

a = ocean.arange(24).reshape([4,6])
success("a")

success("a[0::2]")
success("a[:0]")
success("a[2::-1]")
success("a[2::-1e10,3:-100:-2]")
success("a[1:3,2:-1]")


fail("a[2::-1e200,3:-100:-2]")

