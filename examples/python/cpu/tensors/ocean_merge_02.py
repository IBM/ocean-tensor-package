import pyOcean_cpu as ocean

a = ocean.asTensor(range(8),ocean.float).reshape([1,8])
a = a.broadcastTo([6,8])
print(a)

v = a.split(0,[0,2,3,1])
for i in range(4) :
   print("\n------ Slice %d ------" % (i+1))
   print(v[i])

b = ocean.merge(v,0)
print(b)

print("\n------ Allowed case #1 ------")
ocean.merge(v,0,a)
print(a)

print("\n------ Allowed case #2 ------")
b = ocean.tensor(a.size, a.strides, a.dtype)
b.fill(0)
ocean.merge(v,0,b)
print(b)

print("\n------ Non-allowed case #1 ------")
v[-1] = v[-1].clone()
try :
   ocean.merge(v,0,a)
except :
   print("Expected error: merging along zero-strided dimension")


print("\n------ Non-allowed case #2 ------")
try :
   ocean.merge(ocean.tensor(b.size),0,b)
except :
   print("Expected error: cannot merge non-zero strides slice into zero-strided dimension")
