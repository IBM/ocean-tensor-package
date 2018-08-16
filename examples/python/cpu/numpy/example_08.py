import pyOcean_cpu as ocean
import numpy as np
import pyOceanNumpy

print(ocean.asTensor(np.int8(1)))
print(ocean.asTensor(np.int16(2)))
print(ocean.asTensor(np.int32(3)))
print(ocean.asTensor(np.int64(4)))

print(ocean.asTensor(np.uint8(5)))
print(ocean.asTensor(np.uint16(6)))
print(ocean.asTensor(np.uint32(7)))
print(ocean.asTensor(np.uint64(8)))

print(ocean.asTensor(np.float16(9.0)))
print(ocean.asTensor(np.float32(10.0)))
print(ocean.asTensor(np.float64(11.0)))

print(ocean.asTensor(np.complex64(12+13j)))
print(ocean.asTensor(np.complex128(14+15j)))


