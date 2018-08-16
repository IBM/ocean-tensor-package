import pyOcean_cpu as ocean
import ctypes

t = ocean.tensor([3,4])

FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
ptr = ctypes.cast(t.ptr, FLOAT_PTR)

for i in range(12) :
   ptr[i] = i

print(t)

