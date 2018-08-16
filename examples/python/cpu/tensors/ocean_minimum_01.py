import pyOcean_cpu as ocean

# NaN + 1j
x = ocean.chalf(1j)
x.real = ocean.nan

# 3 + NaN*j
y = ocean.chalf(1)
y.imag = ocean.nan

print([x,y])

print(ocean.minimum(ocean.half([ocean.nan, 2])))
print(ocean.minimum(ocean.half([1,ocean.nan,2])))
print(ocean.minimum(ocean.half([2,ocean.nan,1])))

print("========= Complex with NaN =========")
print(ocean.minimum(ocean.chalf([x, 2+1j])))
print(ocean.minimum(ocean.chalf([y, 0+2j])))
print(ocean.minimum(ocean.chalf([y, 1+3j])))
print(ocean.minimum(ocean.chalf([y, 2+4j])))
print(ocean.minimum(ocean.chalf([x, y])))
print(ocean.minimum(ocean.chalf([y, x])))


print("========= Complex =========")
print(ocean.minimum(ocean.chalf([2+1j, 1+0j])))
print(ocean.minimum(ocean.chalf([2+1j, 1+1j])))
print(ocean.minimum(ocean.chalf([2+1j, 1+2j])))
print(ocean.minimum(ocean.chalf([1+1j, 1+2j])))
print(ocean.minimum(ocean.chalf([1+2j, 1+2j])))
print(ocean.minimum(ocean.chalf([1+4j, 1+3j])))

