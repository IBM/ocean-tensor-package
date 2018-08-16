import ocean

a = ocean.arange(9,ocean.cfloat,ocean.gpu[0]).reshape([3,3])
b = ocean.arange(9,ocean.cfloat,ocean.gpu[0]).reshape([3,3])  

print(a)

alpha = ocean.asTensor([1.0,2j,1+2j],ocean.chalf,ocean.gpu[0]).reshape([1,1,3])
print(alpha)

print(ocean.gemm(alpha,a,b))

