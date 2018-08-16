import pyOcean_cpu as ocean

a = ocean.arange(3*5*3, ocean.int8).reshape([3,5,3]);
a %= 12;

print(a)
print(ocean.nnz(a))
print(ocean.nnz(a,0,True))
print(ocean.nnz(a,1,True))
print(ocean.nnz(a,2,True))
print(ocean.nnz(a,[0,1],True))

