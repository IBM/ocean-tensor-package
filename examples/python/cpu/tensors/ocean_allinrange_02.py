import pyOcean_cpu as ocean

a = ocean.float([1.0])
b = ocean.cdouble(1.0+0.01j)
c = ocean.cdouble(1.0-0.01j)

print([ocean.allLT(a,b), True])
print([ocean.allLT(a,c), False])
print([ocean.allGT(a,b), False])
print([ocean.allGE(a,c), True])

