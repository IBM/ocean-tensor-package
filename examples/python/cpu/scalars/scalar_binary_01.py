## Scalar addition, subtraction, multiplication, and division
import pyOcean_cpu as ocean

def run(cmd) :
   result = eval(cmd)
   print("%s = %s (%s)" % (cmd, str(result), result.dtype.name))

print("\n# Scalar addition")
run("ocean.int8(9) + 3")
run("3 + ocean.uint8(200)")
run("ocean.uint8(100) + ocean.uint8(200)")
run("ocean.int8(10) + ocean.chalf(1+2j)")

print("\n# Scalar subtraction")
run("ocean.int8(9) - 3")
run("3 - ocean.uint8(200)")
run("ocean.uint8(100) - ocean.uint8(200)")
run("ocean.int8(10) - ocean.chalf(1+2j)")

print("\n# Scalar multiplication")
run("ocean.int8(9) * 3")
run("3 * ocean.uint8(200)")
run("ocean.uint8(100) * ocean.uint8(200)")
run("ocean.int8(10) * ocean.chalf(1+2j)")

print("\n#Scalar division")
run("ocean.int8(9) / 3")
run("3 / ocean.uint8(200)")
run("ocean.uint8(100) / ocean.uint8(200)")
run("ocean.int8(10) / ocean.chalf(1+2j)")
run("ocean.chalf(10) / ocean.chalf(1+2j)")

