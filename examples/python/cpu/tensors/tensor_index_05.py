## Indexing with ellipsis #1
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

a = ocean.tensor([])
a.fill(3)

success("a");
success("a[...]")

fail("a[1]")
