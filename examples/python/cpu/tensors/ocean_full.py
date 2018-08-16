import pyOcean_cpu as ocean

a = ocean.full([1,2],1)
print(a)

a = ocean.full([1,2],[2.])
print(a)

a = ocean.full([1,2],ocean.asTensor(3,ocean.double))
print(a)

a = ocean.full([1,2],ocean.asTensor(4,ocean.double),ocean.int8)
print(a)

a = ocean.full([1,2],ocean.int16(5),ocean.cpu)
print(a)

a = ocean.full([1,2],ocean.int16(6),ocean.float,ocean.cpu)
print(a)


