import pyOcean_cpu as ocean

print(ocean.tensor([]).isScalar())
print(ocean.tensor([2]).isScalar())
print(ocean.tensor([1]).isScalar())
