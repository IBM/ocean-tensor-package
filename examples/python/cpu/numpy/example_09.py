import pyOcean_cpu as ocean
import numpy as np
import pyOceanNumpy

print(ocean.asTensor([np.int8(1), np.int16(2), np.int32(3), np.int64(4)]))

print(ocean.asTensor([np.uint8(5), np.uint16(6), np.uint32(7), np.uint64(8)]))

print(ocean.asTensor([np.float16(9.0), np.float32(10.0), np.float64(11.0)]))

print(ocean.asTensor([np.complex64(12+13j), np.complex128(14+15j)]))

