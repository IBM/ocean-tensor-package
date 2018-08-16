## Basic tensor properties
import pyOcean_cpu as ocean

T = ocean.tensor([2,3],ocean.float)
T.storage.zero()
print(T)
print(T.storage)

print(T.device)
print(T.dtype)
print(T.ndims)
print(T.size)
print(T.strides)
print(T.offset)
print(T.device)
print(T.elemsize)
print(T.refcount)
print(T.footer)

