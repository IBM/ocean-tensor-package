import pyOcean_cpu as ocean

A = ocean.asTensor([[0,1],[2,3]],'r',ocean.float)
b = ocean.asTensor([1,2],ocean.float)
B = b.reshape([2,1]);
c = ocean.tensor([2],ocean.float);
C = ocean.tensor([2,1],ocean.float);

print(A)
print(b)
print(B)


print("\n====== A*b ======");
print(ocean.gemm(1,A,b))

print("\n====== A*B ======");
print(ocean.gemm(1,A,B))

print("\n====== c = A*b ======");
ocean.gemm(1,A,b,0,c); print(c)

print("\n====== C = A*b ======");
ocean.gemm(1,A,b,0,C); print(C)

print("\n====== c = A*B ======");
ocean.gemm(1,A,B,0,c); print(c)

print("\n====== C = A*B ======");
ocean.gemm(1,A,B,0,C); print(C)

