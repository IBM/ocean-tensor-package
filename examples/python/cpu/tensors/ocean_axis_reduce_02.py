import pyOcean_cpu as ocean

# This should run on a single thread
a = ocean.ones([10,10],ocean.double)
print(ocean.sum(a,0))

# Each thread should reduce the result for its outputs
a = ocean.ones([100,50],ocean.double)
print(ocean.sum(a,0))

# All threads should jointly reduce each output
a = ocean.ones([3,10000],ocean.double)
print(ocean.sum(a,1))

