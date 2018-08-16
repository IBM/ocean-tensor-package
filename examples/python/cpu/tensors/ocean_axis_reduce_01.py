import pyOcean_cpu as ocean

a = ocean.arange(3*4*5,ocean.int8).reshape([3,4,5]);

print(ocean.sum(a,2))
print(ocean.sum(a,2,True))

b = ocean.tensor([3,4],ocean.double);
b.byteswap()
ocean.sum(a,2,b)
print(b)

b = ocean.tensor([3,4,1],ocean.int64);
ocean.sum(a,2,True,b)
print(b)



