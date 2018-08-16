import pyOcean_cpu as ocean

a = ocean.asTensor([-128,-1,0,1,127],ocean.int8)

print(-a)

