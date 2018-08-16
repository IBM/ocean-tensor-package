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

a = ocean.arange(2*6*2*3).reshape([2,6,2,3])
success("a[...]")

success("a[...,2]")
success("a[...,-1,:]")
success("a[1,...]")
success("a[1,None,...]")

success("a[1,1,1,1,...]")

fail("a[1,1,1,1,...,1]")

