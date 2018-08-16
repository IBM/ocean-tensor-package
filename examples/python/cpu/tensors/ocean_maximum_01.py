import pyOcean_cpu as ocean

# NaN + 1j
x = ocean.chalf(1j)
x.real = ocean.nan

# 3 + NaN*j
y = ocean.chalf(3)
y.imag = ocean.nan

print([x,y])

print(ocean.maximum(ocean.half([ocean.nan, 2])))
print(ocean.maximum(ocean.half([1,ocean.nan,2])))
print(ocean.maximum(ocean.half([2,ocean.nan,1])))

print("========= Complex with NaN =========")
print(ocean.maximum(ocean.chalf([x, 2+1j])))
print(ocean.maximum(ocean.chalf([y, 1+2j])))
print(ocean.maximum(ocean.chalf([y, 1+3j])))
print(ocean.maximum(ocean.chalf([y, 1+4j])))
print(ocean.maximum(ocean.chalf([x,y])))
print(ocean.maximum(ocean.chalf([y,x])))


print("========= Complex =========")
print(ocean.maximum(ocean.chalf([1+1j, 2+0j])))
print(ocean.maximum(ocean.chalf([1+1j, 2+1j])))
print(ocean.maximum(ocean.chalf([1+1j, 2+2j])))
print(ocean.maximum(ocean.chalf([1+1j, 1+0j])))
print(ocean.maximum(ocean.chalf([1+2j, 1+2j])))
print(ocean.maximum(ocean.chalf([1+2j, 1+3j])))

