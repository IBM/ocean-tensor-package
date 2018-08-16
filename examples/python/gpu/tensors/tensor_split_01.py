import ocean

a = ocean.tensor([5,6])
a.copy(range(a.nelem))

print("-----------------------------------")
print("a")
print("-----------------------------------")
print(a)

print("\n-----------------------------------")
print("a.split(0,[ocean.cpu, ocean.gpu[0], ocean.cpu])")
print("-----------------------------------")
v = a.split(0,[ocean.cpu, ocean.gpu[0], ocean.cpu])
for tensor in v :
   print(tensor)

print("\n-----------------------------------")
print("a.split(1,[1,5,0])")
print("-----------------------------------")
v = a.split(1,[1,5,0])
for tensor in v :
   print(tensor)

print("\n-----------------------------------")
print("a.split(0,[ocean.gpu[0],ocean.cpu],[1,4])")
print("-----------------------------------")
v = a.split(0,[ocean.gpu[0],ocean.cpu],[1,4])
for tensor in v :
   print(tensor)

