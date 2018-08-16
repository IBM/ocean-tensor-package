import pyOcean_cpu as ocean

a = ocean.asTensor([[1,2],[3,4]],'r',ocean.float)
b = ocean.arange(6,ocean.float).reshape([2,3])

print(a)
print(b)
print(ocean.gemm(ocean.float(3),a,b))

alpha = ocean.asTensor([1,2,3],ocean.float).reshape([1,1,3]);
print(ocean.gemm(alpha,a,b))
print(ocean.gemm(ocean.float(1),a.T,b))

