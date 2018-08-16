import pyOcean_cpu as ocean

# Create an index based on A
A = ocean.int64([1,2,3])
idx = ocean.index[A]

# Make sure that the tensor is properly detached
A.fill(5)

# Apply the index
print(idx)
b = ocean.arange(10)
print(b[idx])

