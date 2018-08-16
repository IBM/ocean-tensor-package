import pyOcean_cpu as ocean

a = ocean.asTensor([1+2j,2+3j],'r', ocean.cfloat);
b = ocean.asTensor([1,2+1j], ocean.cfloat);

print(a)
print(ocean.conj(a))
print(b)

print(ocean.gemm(1,a,b))
print(ocean.gemm(1,a.T,'C',b))
print(ocean.gemm(1,a.T,'C',b.T,'C'))
print(ocean.gemm(1,a,b.T,'C'))

