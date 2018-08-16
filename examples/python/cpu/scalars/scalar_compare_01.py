import pyOcean_cpu as ocean

a = ocean.double(8)
print(a == 8)
print(a >= 8)
print(a <= 8)
print(a > 7)
print(a < 9)


print(a != 8)
print(a < 8)
print(a > 8)
print(a > 10)


print("HALF")
a = ocean.half(9)
b = ocean.half(8)
print(a == b)
print(b > 10)
print(b < 10)
