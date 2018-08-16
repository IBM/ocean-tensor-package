import pyOcean_cpu as ocean

import sys

def exceptionMsg() :
   print("Expected error: %s" % str(sys.exc_info()[1]))

def failTest(command) :
   try :
      eval(command)
   except :
      exceptionMsg()


a = ocean.tensor([2,3,4,5])
failTest("a.permuteAxes(range(100))")
failTest("a.permuteAxes([1,2,3,4,5])")
failTest("a.permuteAxes([-1,1,2,3])")
failTest("a.permuteAxes([1,2,3,20])")
failTest("a.permuteAxes([1,2,3,1])")

