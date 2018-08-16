import pyOcean_cpu as ocean

a = ocean.asTensor([[0,1],[2,3]],'r',ocean.float)
b = ocean.asTensor([1,2],ocean.float)
c = ocean.tensor([2],[128],ocean.float);

print(a)
print(b)
ocean.gemm(1,a,b,0,c)
print(c)


print("\n----------------------------------------")
a = ocean.asTensor([1.,2,3]).reshape([3,1]);
b = ocean.asTensor([2.,3,4],'r');

c = ocean.tensor([3,3],a.dtype);
print(a)
print(b)
ocean.gemm(1,a,b,0,c);
print(c)
#print(ocean.gemm(1,a,b))

