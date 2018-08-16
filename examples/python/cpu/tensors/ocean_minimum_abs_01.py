import pyOcean_cpu as ocean

# NaN + 1j
x = ocean.chalf(1j)
x.real = ocean.nan

# 3 + NaN*j
y = ocean.chalf(1)
y.imag = ocean.nan

print([x,y])

print(ocean.minimumAbs(ocean.half([ocean.nan, 2])))
print(ocean.minimumAbs(ocean.half([1,ocean.nan,2])))
print(ocean.minimumAbs(ocean.half([-2,ocean.nan,1])))

print("========= Complex with NaN =========")
print(ocean.minimumAbs(ocean.chalf([x, 3+4j])))
print(ocean.minimumAbs(ocean.chalf([y, 0+2j])))
print(ocean.minimumAbs(ocean.chalf([x,y])))

print("========= Complex =========")
print(ocean.minimumAbs(ocean.chalf([3+4j, 1+0j])))
print(ocean.minimumAbs(ocean.chalf([1+0j, 3+4j])))

