import pyOcean_cpu as ocean

a = ocean.tensor([3])
b = a.shallowCopy()
print(a.obj == b.obj)
print(a.storage.obj == b.storage.obj)

