import pyOcean_cpu as ocean

print("Initialization: a = 17 (1001), b = 3 (011)")
b = ocean.int8(3)

print("\n------ a & b, ocean.bitwiseAnd(a,b), a &= b ------")
a = ocean.int8(17)
print(a & b)
print(ocean.bitwiseAnd(a,b))
a &= b
print(a)

print("\n------ a | b, ocean.bitwiseOr(a,b), a |= b ------")
a = ocean.int8(17)
print(a | b)
print(ocean.bitwiseOr(a,b))
a |= b
print(a)

print("\n------ a ^ b, ocean.bitwiseXor(a,b), a ^= b ------")
a = ocean.int8(17)
print(a ^ b)
print(ocean.bitwiseXor(a,b))
a ^= b
print(a)

