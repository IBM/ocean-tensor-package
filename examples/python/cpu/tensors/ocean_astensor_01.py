## ocean.asTensor - Empty tensors
import pyOcean_cpu as ocean

print(ocean.asTensor([]))
print(ocean.asTensor([[[]]],ocean.int32))
print(ocean.asTensor([[],[],[]]))
