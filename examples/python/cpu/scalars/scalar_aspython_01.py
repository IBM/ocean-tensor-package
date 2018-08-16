import pyOcean_cpu as ocean

a = ocean.bool(0)
print(a.asPython())

a = ocean.int16(1)
print(a.asPython())

a = ocean.float(2)
print(a.asPython())

a = ocean.chalf(3+4j)
print(a.asPython())
