import pyOcean_cpu as ocean

# NaN + 1j
x = ocean.chalf(1j)
x.real = ocean.nan

# 3 + NaN*j
y = ocean.chalf(1)
y.imag = ocean.nan

print([x,y])

print(ocean.maximumAbs(ocean.half([ocean.nan, 2])))
print(ocean.maximumAbs(ocean.half([1,ocean.nan,2])))
print(ocean.maximumAbs(ocean.half([-2,ocean.nan,1])))

print("========= Complex with NaN =========")
print(ocean.maximumAbs(ocean.chalf([x, 3+4j])))
print(ocean.maximumAbs(ocean.chalf([y, 0+2j])))
print(ocean.maximumAbs(ocean.chalf([x,y])))

print("========= Complex =========")
print(ocean.maximumAbs(ocean.chalf([3+4j, 1+0j])))
print(ocean.maximumAbs(ocean.chalf([1+0j, 3+4j])))

