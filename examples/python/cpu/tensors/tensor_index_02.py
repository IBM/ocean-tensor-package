## Range indexing #1
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

success("a[1,2]")
success("a[:]")
success("a[::]")
success("a[::-1]")
success("a[:,::-2]")
success("a[:2:,::-5]")
success("a[1,::-6]")


fail("a[::0]")

fail("a[1+2j]")
fail("a[:1+2j]")
fail("a[::1+2j]")

fail("a[ocean.nan]")
fail("a[:ocean.nan]")
fail("a[::ocean.nan]")

