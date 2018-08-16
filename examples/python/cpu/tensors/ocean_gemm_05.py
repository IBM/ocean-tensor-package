import pyOcean_cpu as ocean

a = ocean.asTensor([[0,1],[2,3]],'r',ocean.float)
b = ocean.asTensor([1,2],ocean.float)
c = ocean.tensor([2,1,3],ocean.float);

ocean.gemm(ocean.float(1),a,b,0,c)
print(c)

