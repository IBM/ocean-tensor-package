## Scalar indexing
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

a = ocean.arange(10)
success("a")

success("a[0]")
success("a[1]")
success("a[9]")
success("a[-1]")
success("a[-10]")

fail("a[10]")
fail("a[-11]")
fail("a[1+2j]")
fail("a[ocean.nan]")
fail("a[1e200]")
fail("a[ocean.inf]")

