## Scalar in-place addition, subtraction, multiplication, and division
import pyOcean_cpu as ocean

def iadd(cmd1,cmd2) :
   a = eval(cmd1)
   b = eval(cmd2)
   s = "%s += %s" % (cmd1, cmd2)
   a += b
   print("%-30s ===> %s (%s)" % (s, str(a), a.dtype.name))

def isub(cmd1,cmd2) :
   a = eval(cmd1)
   b = eval(cmd2)
   s = "%s -= %s" % (cmd1, cmd2)
   a -= b
   print("%-30s ===> %s (%s)" % (s, str(a), a.dtype.name))

def imul(cmd1,cmd2) :
   a = eval(cmd1)
   b = eval(cmd2)
   s = "%s *= %s" % (cmd1, cmd2)
   a *= b
   print("%-30s ===> %s (%s)" % (s, str(a), a.dtype.name))

def idiv(cmd1,cmd2) :
   a = eval(cmd1)
   b = eval(cmd2)
   s = "%s /= %s" % (cmd1, cmd2)
   a /= b
   print("%-30s ===> %s (%s)" % (s, str(a), a.dtype.name))



print("# Inplace addition")
iadd("ocean.int8(3)", "4")

print("\n# Inplace subtraction")
isub("ocean.uint8(3)", "4.")

print("\n# Inplace multiplication")
imul("ocean.chalf(3+2j)","4+2j")

print("\n# Inplace division")
idiv("ocean.chalf(3+2j)","4+2j")


