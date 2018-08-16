import pyOcean_cpu as ocean

a = ocean.tensor([6,8])
a.copy(range(a.nelem));

v = a.split(0,3)

print("MERGE INTO NEW TENSOR")
print(ocean.merge(v,0))

print("\nMERGE INTO THE SAME TENSOR")
ocean.merge(v,0,a)
print(a)

print("\nMERGE INTO NEW TENSOR - REVERSE ORDER")
print(ocean.merge([v[2],v[1],v[0]],0))


print("\nMERGE INTO THE SAME TENSOR - REVERSE ORDER")
ocean.merge([v[2],v[1],v[0]],0,a)
print(a)
