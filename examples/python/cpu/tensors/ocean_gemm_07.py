import pyOcean_cpu as ocean

a = ocean.asTensor([[0,1,2.],[3,4,5]]);
b = ocean.arange(12,ocean.double).reshape([3,4]);

print(a.T)
print(b)
print(ocean.gemm(1,a,'T',b))
print(ocean.gemm(1,a,a,'T'))

