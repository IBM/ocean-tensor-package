import pyOcean_cpu as ocean

a = ocean.asTensor([[1,2,ocean.nan],[3+4j,4,5]])
print(a)

print(ocean.sum(a,0))
print(ocean.sumNaN(a,0))
print(ocean.sumAbs(a,0))
print(ocean.sumAbsNaN(a,0))
print(ocean.prod(a,0))
print(ocean.prodNaN(a,0))

