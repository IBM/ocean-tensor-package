import pyOcean_cpu as ocean

print(ocean.arange(10))
print(ocean.arange(2,10))
print(ocean.arange(2,10,3,ocean.int8))
print(ocean.arange(10,2,-3,ocean.half))

print(ocean.arange(10.))
print(ocean.arange(2.,10))
print(ocean.arange(2,10,3.,ocean.int8))
print(ocean.arange(10,2.,-3,ocean.half))

print("\n--------- Non-integer step ---------")
print(ocean.arange(2,5,0.3))

print("\n--------- Single element ---------")
print(ocean.arange(2,5,10))
print(ocean.arange(2,5,ocean.inf))

print("\n--------- Empty ---------")
print(ocean.arange(5,2))
print(ocean.arange(2,5,-10))
print(ocean.arange(2,5,-ocean.inf))

print(ocean.arange(5,2.))
print(ocean.arange(2,5,-10.))
print(ocean.arange(2,5,-ocean.inf))

