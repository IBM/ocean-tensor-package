import pyOcean_cpu as ocean

s = ocean.cdouble(3+4j)
print(s)

print(s.asPython())

print(s.imag.asPython())
print(s.real.asPython())
print(int(s.real))
print(float(s.real))

