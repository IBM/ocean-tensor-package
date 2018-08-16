import pyOcean_cpu as ocean

a = ocean.tensor([5,6])
a.T.copy(range(a.nelem))

print("---------------------------------------")
print("a")
print("---------------------------------------")
print(a)

print("\n---------------------------------------")
print("a.split(1,[1,5,0])")
print("---------------------------------------")
v = a.split(1,[1,5,0])
for tensor in v :
   print(tensor)

print("\n---------------------------------------")
print("a.split(0,[ocean.cpu, ocean.cpu],[1,4])")
print("---------------------------------------")
v = a.split(0,[ocean.cpu, ocean.cpu],[1,4])
for tensor in v :
   print(tensor)

