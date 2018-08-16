import pyOcean_cpu as ocean

a = ocean.asTensor([[1+2j,2+1j],[3+0j,4+2j]],'r',ocean.cfloat)
b = ocean.asTensor([1+1j,2-3j],ocean.cfloat)

print(ocean.gemm(ocean.float(1),a,b))

